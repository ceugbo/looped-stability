"""Training, evaluation and the experiment matrix for the looped-model injection study.

Usage (from the p4_looped folder, on a CUDA GPU):
    python data.py --out_dir data                       # tokenise once
    python train.py --exp stage1 --data_dir data --run_dir results
    python train.py --exp stage3 --data_dir data --run_dir results
    python train.py --exp stress --data_dir data --run_dir results

Each run saves <run_dir>/<name>.json (config, logs, final evaluation) and a resumable
checkpoint <run_dir>/<name>.pt (not tracked by git).
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch

from data import get_batch, load_data, make_val_set
from model import Config, LoopedLM, sample_loops


@torch.no_grad()
def evaluate(model, val_batches, Ts):
    """Validation loss / perplexity / accuracy at fixed loop counts (same depth for every sequence)."""
    model.eval()
    out = {}
    for T in Ts:
        tl = ta = 0.0
        for x, y in val_batches:
            with torch.autocast("cuda", dtype=torch.float16):
                logits, loss, _ = model(x, n_loops=T, targets=y)
            tl += loss.item()
            ta += (logits.argmax(-1) == y).float().mean().item()
        n = len(val_batches)
        out[T] = dict(val_loss=tl / n, val_ppl=math.exp(min(tl / n, 20)), val_acc=ta / n)
    model.train()
    return out


@torch.no_grad()
def state_norms(model, val_batches, T=16):
    """RMS of the loop state after each of T loops, on one validation batch."""
    model.eval()
    x, _ = val_batches[0]
    with torch.autocast("cuda", dtype=torch.float16):
        _, _, norms = model(x, n_loops=T, return_norms=True)
    model.train()
    return norms


def make_optimizer(model, lr, wd=0.1):
    decay = [p for p in model.parameters() if p.dim() >= 2]      # weight matrices, embeddings
    no_decay = [p for p in model.parameters() if p.dim() < 2]    # norm scales, A / dt parameters
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": wd},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=lr, betas=(0.9, 0.95))


def lr_at(step, steps, lr, warm=100, min_frac=0.1):
    if step < warm:                                              # linear warmup
        return lr * (step + 1) / warm
    p = (step - warm) / max(1, steps - warm)                     # progress through the rest
    return lr * (min_frac + (1 - min_frac) * 0.5 * (1 + math.cos(math.pi * p)))   # cosine decay


def run(name, train_data, val_batches, run_dir, inject="constrained", k_grad=2, mean_loops=6,
        steps=4000, lr=6e-4, batch=16, seed=0, eval_every=500, log_every=20,
        final_Ts=(1, 2, 4, 6, 8, 12, 16, 24)):
    os.makedirs(run_dir, exist_ok=True)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    cfg = Config(inject=inject)
    model = LoopedLM(cfg).cuda()
    opt = make_optimizer(model, lr)
    scaler = torch.amp.GradScaler("cuda")

    log, evals = [], []
    ema, spikes, nonfinite, start = None, 0, 0, 0

    ckpt = os.path.join(run_dir, f"{name}.pt")
    if os.path.exists(ckpt):                                     # resume after a disconnect
        s = torch.load(ckpt, map_location="cuda")
        model.load_state_dict(s["model"])
        opt.load_state_dict(s["opt"])
        scaler.load_state_dict(s["scaler"])
        start, log, evals = s["step"], s["log"], s["evals"]
        ema, spikes, nonfinite = s["ema"], s["spikes"], s["nonfinite"]
        rng = np.random.default_rng(seed + start)                # batches differ from an uninterrupted run
        print(f"resumed {name} at step {start}")

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()

    for step in range(start, steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, steps, lr)

        x, y = get_batch(train_data, batch, cfg.ctx, rng)
        n = sample_loops(batch, mean=mean_loops, lo=1, hi=2 * mean_loops)

        with torch.autocast("cuda", dtype=torch.float16):
            _, loss, _ = model(x, n_loops=n, k_grad=k_grad, targets=y)

        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()   # norm before clipping
        scaler.step(opt)
        scaler.update()

        l = loss.item()
        if not math.isfinite(l):
            nonfinite += 1
        else:
            if ema is not None and step > 200 and l > 1.5 * ema:
                spikes += 1                                      # sudden jump above 1.5x running average
            ema = l if ema is None else 0.98 * ema + 0.02 * l

        if step % log_every == 0:
            log.append(dict(step=step, loss=l, grad_norm=gn, clipped=int(gn > 1.0),
                            spikes=spikes, nonfinite=nonfinite,
                            max_A=model.inj()[0].max().item(),
                            lr=lr_at(step, steps, lr), time=time.time() - t0,
                            mem_gb=torch.cuda.max_memory_allocated() / 1e9))

        if (step + 1) % eval_every == 0:
            ev = evaluate(model, val_batches, [4, 8])
            evals.append(dict(step=step + 1, evals=ev, state_norms=state_norms(model, val_batches, 16)))
            print(f'{name} step {step + 1} | train {l:.3f} | val@8 {ev[8]["val_loss"]:.3f} '
                  f'| gn {gn:.2f} | spikes {spikes} | nonfinite {nonfinite} | {time.time() - t0:.0f}s')
            torch.save(dict(model=model.state_dict(), opt=opt.state_dict(),
                            scaler=scaler.state_dict(), step=step + 1, log=log,
                            evals=evals, ema=ema, spikes=spikes, nonfinite=nonfinite), ckpt)

    final = evaluate(model, val_batches, list(final_Ts))
    result = dict(name=name, inject=inject, k_grad=k_grad, mean_loops=mean_loops,
                  seed=seed, steps=steps, lr=lr, spikes=spikes, nonfinite=nonfinite,
                  train_time_s=time.time() - t0,                 # covers only steps since the last resume
                  peak_mem_gb=torch.cuda.max_memory_allocated() / 1e9,
                  final=final, final_state_norms=state_norms(model, val_batches, 16),
                  evals=evals, log=log)
    with open(os.path.join(run_dir, f"{name}.json"), "w") as f:
        json.dump(result, f)
    print(name, "done |", {T: round(v["val_loss"], 3) for T, v in final.items()})
    return result


def experiment_matrix(exp):
    """(name, kwargs) pairs for each experiment, as run for the report."""
    runs = []
    if exp in ("stage1", "all"):          # injection ablation, two seeds, lr 6e-4, 6000 steps
        for seed in (0, 1):
            for inj in ("free", "constrained"):
                runs.append((f"s1_{inj}_seed{seed}",
                             dict(inject=inj, k_grad=2, steps=6000, seed=seed, lr=6e-4)))
    if exp in ("stage3", "all"):          # gradient truncation depth (k_grad=2 is the stage-1 constrained run)
        for k in (1, 4):
            runs.append((f"s3_k{k}", dict(inject="constrained", k_grad=k, steps=6000, seed=0, lr=6e-4)))
    if exp in ("stress", "all"):          # higher learning rate, shorter schedule
        for inj in ("free", "constrained"):
            runs.append((f"stress_{inj}_lr0.002",
                         dict(inject=inj, k_grad=2, steps=3000, seed=0, lr=2e-3)))
    return runs


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--exp", choices=["stage1", "stage3", "stress", "all"], required=True)
    p.add_argument("--data_dir", default="data")
    p.add_argument("--run_dir", default="results")
    args = p.parse_args()

    assert torch.cuda.is_available(), "These scripts expect a CUDA GPU (fp16 autocast)."
    train_data, val_data = load_data(args.data_dir)
    val_batches = make_val_set(val_data)

    for name, kwargs in experiment_matrix(args.exp):
        if os.path.exists(os.path.join(args.run_dir, f"{name}.json")):
            print("skipping finished run", name)
            continue
        run(name, train_data, val_batches, args.run_dir, **kwargs)

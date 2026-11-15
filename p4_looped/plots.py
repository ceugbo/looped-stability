"""Make the three comparison figures from the saved result files.

    python plots.py --run_dir results --fig_dir ../figures
"""
import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main(run_dir, fig_dir):
    os.makedirs(fig_dir, exist_ok=True)

    def load(name):
        with open(os.path.join(run_dir, f"{name}.json")) as f:
            return json.load(f)

    # Figure 1: validation loss against loop count at evaluation (free vs constrained, seed 0)
    fig, ax = plt.subplots(figsize=(6, 4))
    for label, name in [("free", "s1_free_seed0"), ("constrained", "s1_constrained_seed0")]:
        r = load(name)
        Ts = sorted(int(t) for t in r["final"])
        ax.plot(Ts, [r["final"][str(t)]["val_loss"] for t in Ts], marker="o", label=label)
    ax.set_xlabel("loops at evaluation (T)")
    ax.set_ylabel("validation loss")
    ax.set_title("Validation loss vs loop count (lr 6e-4)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig1_loss_vs_loops.png"), dpi=200)
    plt.close(fig)

    # Figure 2: loop-state RMS across iterations (linear axis)
    fig, ax = plt.subplots(figsize=(6, 4))
    for label, name in [("free", "s1_free_seed0"), ("constrained", "s1_constrained_seed0")]:
        n = load(name)["final_state_norms"]
        ax.plot(range(1, len(n) + 1), n, marker="o", label=label)
    ax.set_xlabel("loop iteration")
    ax.set_ylabel("state RMS")
    ax.set_title("Loop state across iterations (lr 6e-4)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig2_state_norms.png"), dpi=200)
    plt.close(fig)

    # Figure 3: effect of gradient truncation depth (constrained, seed 0)
    fig, ax = plt.subplots(figsize=(6, 4))
    for label, name in [("k_grad=1", "s3_k1"), ("k_grad=2", "s1_constrained_seed0"), ("k_grad=4", "s3_k4")]:
        ev = load(name)["evals"]
        ax.plot([e["step"] for e in ev], [e["evals"]["8"]["val_loss"] for e in ev], marker="o", label=label)
    ax.set_xlabel("training step")
    ax.set_ylabel("validation loss (T=8)")
    ax.set_title("Effect of gradient truncation depth")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig3_kgrad.png"), dpi=200)
    plt.close(fig)

    print("saved figures to", fig_dir)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", default="results")
    p.add_argument("--fig_dir", default="../figures")
    args = p.parse_args()
    main(args.run_dir, args.fig_dir)

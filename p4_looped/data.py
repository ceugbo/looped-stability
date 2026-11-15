"""Data preparation and batching.

prepare_data()  tokenises the first 10% of TinyStories with the GPT-2 tokenizer and writes
                train.bin / val.bin (uint16 token streams; the last 1% of tokens is validation).
get_batch()     samples random windows; y is x shifted by one token.
make_val_set()  builds the fixed validation batches shared by every run.
"""
import os
import numpy as np
import torch


def prepare_data(out_dir, split="train[:10%]"):
    from datasets import load_dataset
    from transformers import AutoTokenizer

    os.makedirs(out_dir, exist_ok=True)
    tok = AutoTokenizer.from_pretrained("gpt2")
    eos = tok.eos_token_id
    ds = load_dataset("roneneldan/TinyStories", split=split)

    def tokenize(batch):
        ids = tok(batch["text"])["input_ids"]
        return {"ids": [x + [eos] for x in ids]}

    tokenized = ds.map(tokenize, batched=True, remove_columns=ds.column_names, num_proc=2)
    arr = np.concatenate([np.array(x, dtype=np.uint16) for x in tokenized["ids"]])
    print(f"{len(arr) / 1e6:.1f}M tokens")

    n_val = int(0.01 * len(arr))
    arr[:-n_val].tofile(os.path.join(out_dir, "train.bin"))
    arr[-n_val:].tofile(os.path.join(out_dir, "val.bin"))


def load_data(data_dir):
    train = np.memmap(os.path.join(data_dir, "train.bin"), dtype=np.uint16, mode="r")
    val = np.memmap(os.path.join(data_dir, "val.bin"), dtype=np.uint16, mode="r")
    return train, val


def get_batch(data, batch_size, ctx, rng, device="cuda"):
    ix = rng.integers(0, len(data) - ctx - 1, batch_size)
    x = np.stack([data[i:i + ctx] for i in ix]).astype(np.int64)
    y = np.stack([data[i + 1:i + 1 + ctx] for i in ix]).astype(np.int64)
    return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)


def make_val_set(val, n_batches=16, batch_size=16, ctx=256, seed=1234, device="cuda"):
    """Fixed validation windows: identical for every run, so results are comparable."""
    rng = np.random.default_rng(seed)
    return [get_batch(val, batch_size, ctx, rng, device) for _ in range(n_batches)]


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default="data")
    args = p.parse_args()
    prepare_data(args.out_dir)

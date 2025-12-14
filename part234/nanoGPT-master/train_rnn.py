import os
import time
import math
import pickle
import argparse
import numpy as np
import torch
from torch.optim import AdamW

from rnn_model import LSTMLanguageModel

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="data/abc_char")
parser.add_argument("--out_dir", type=str, default="out-rnn")
parser.add_argument("--hidden_size", type=int, required=True)
parser.add_argument("--num_layers", type=int, required=True)
parser.add_argument("--dropout", type=float, default=0.1)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--block_size", type=int, default=256)
parser.add_argument("--learning_rate", type=float, default=3e-4)
parser.add_argument("--device", type=str, default="cuda")
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)

with open(os.path.join(args.data_dir, "meta.pkl"), "rb") as f:
    meta = pickle.load(f)

vocab_size = meta["vocab_size"]
print(f"[INFO] vocab_size = {vocab_size}")

train_data = np.memmap(
    os.path.join(args.data_dir, "train.bin"),
    dtype=np.uint16,
    mode="r",
)
val_data = np.memmap(
    os.path.join(args.data_dir, "val.bin"),
    dtype=np.uint16,
    mode="r",
)

def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - args.block_size - 1, (args.batch_size,))
    x = torch.stack([
        torch.from_numpy((data[i:i+args.block_size]).astype(np.int64))
        for i in ix
    ])
    y = torch.stack([
        torch.from_numpy((data[i+1:i+args.block_size+1]).astype(np.int64))
        for i in ix
    ])
    return x.to(args.device), y.to(args.device)

model = LSTMLanguageModel(
    vocab_size=vocab_size,
    hidden_size=args.hidden_size,
    num_layers=args.num_layers,
    dropout=args.dropout,
).to(args.device)

print(f"[INFO] Parameters: {model.num_parameters()/1e6:.2f}M")

optimizer = AdamW(model.parameters(), lr=args.learning_rate)

tokens_per_iter = args.batch_size * args.block_size
num_iters = len(train_data) // tokens_per_iter

print(f"[INFO] tokens / iter = {tokens_per_iter}")
print(f"[INFO] total iters (≈1 epoch) = {num_iters}")

start_time = time.time()
model.train()

for it in range(num_iters):
    xb, yb = get_batch("train")
    logits, loss = model(xb, yb)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if it % 100 == 0:
        elapsed = time.time() - start_time
        print(f"iter {it}: loss {loss.item():.4f}, time {elapsed:.1f}s")

model.eval()
with torch.no_grad():
    losses = []
    for _ in range(100):
        xb, yb = get_batch("val")
        _, loss = model(xb, yb)
        losses.append(loss.item())
    val_loss = sum(losses) / len(losses)

total_time = time.time() - start_time

print("\n===== DONE =====")
print(f"Train loss (last): {loss.item():.4f}")
print(f"Val loss: {val_loss:.4f}")
print(f"Time / epoch (min): {total_time/60:.2f}")

ckpt = {
    "model_state": model.state_dict(),
    "config": vars(args),
    "val_loss": val_loss,
}
torch.save(ckpt, os.path.join(args.out_dir, "ckpt.pt"))

import os
import math
import argparse
import numpy as np
import torch

from model import GPTConfig, GPT

def get_batch(data, block_size, batch_size, device):
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+block_size+1]).astype(np.int64)) for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss(model, data, block_size, batch_size, eval_iters, device):
    model.eval()
    losses = []
    for _ in range(eval_iters):
        X, Y = get_batch(data, block_size, batch_size, device)
        logits, loss = model(X, Y)
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/abc_char", help="Directory containing *.bin + meta.pkl")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--block_size", type=int, default=256)
    ap.add_argument("--eval_iters", type=int, default=200)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    ckpt_path = "/root/autodl-tmp/nanoGPT-master/part4_results/epoch_03/ckpt.pt"
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"ckpt.pt not found: {ckpt_path}")

    val_bin = os.path.join(args.data_dir, "val.bin")
    test_bin = os.path.join(args.data_dir, "test.bin")

    if not os.path.exists(val_bin):
        raise FileNotFoundError(f"val.bin not found: {val_bin}")
    if not os.path.exists(test_bin):
        raise FileNotFoundError(f"test.bin not found: {test_bin}")

    val_data = np.memmap(val_bin, dtype=np.uint16, mode="r")
    test_data = np.memmap(test_bin, dtype=np.uint16, mode="r")

    checkpoint = torch.load(ckpt_path, map_location=args.device)
    model_args = checkpoint["model_args"]
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    model.load_state_dict(checkpoint["model"])
    model.to(args.device)

    val_loss = estimate_loss(model, val_data, args.block_size, args.batch_size, args.eval_iters, args.device)
    test_loss = estimate_loss(model, test_data, args.block_size, args.batch_size, args.eval_iters, args.device)

    val_ppl = math.exp(val_loss)
    test_ppl = math.exp(test_loss)

    print("===== EVAL RESULTS =====")
    print(f"val_loss  = {val_loss:.6f} | val_ppl  = {val_ppl:.3f}")
    print(f"test_loss = {test_loss:.6f} | test_ppl = {test_ppl:.3f}")

if __name__ == "__main__":
    main()

import os
import pickle
import numpy as np
from multiprocessing import Pool, cpu_count

BASE_DIR = os.path.dirname(__file__)
TRAIN_LIST = "../../../data/splits_unique/train.txt"
VAL_LIST   = "../../../data/splits_unique/val.txt"
VOCAB_PATH = "../../../data/vocab_charlevel.txt"

OUT_TRAIN_BIN = os.path.join(BASE_DIR, "train.bin")
OUT_VAL_BIN   = os.path.join(BASE_DIR, "val.bin")
OUT_META      = os.path.join(BASE_DIR, "meta.pkl")

TMP_DIR = os.path.join(BASE_DIR, "tmp_parts")
os.makedirs(TMP_DIR, exist_ok=True)

NUM_WORKERS = min(16, cpu_count())

CHUNK_SIZE = 200        
FLUSH_TOKENS = 500_000  

chars = []
with open(VOCAB_PATH, "r", encoding="utf-8") as f:
    for line in f:
        tok = line.rstrip("\n")
        if tok == r"\n":
            tok = "\n"
        chars.append(tok)

chars = sorted(set(chars))
vocab_size = len(chars)
print(f"[INFO] Loaded char vocab, size = {vocab_size}")

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

def load_file_list(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and os.path.exists(line.strip())]

def encode_worker(args):
    """
    Encode a chunk of files to a temporary binary file.
    """
    chunk_id, paths = args
    out_path = os.path.join(TMP_DIR, f"part_{chunk_id:06d}.bin")

    buf = []
    token_count = 0

    with open(out_path, "wb") as fout:
        for path in paths:
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as fin:
                    for line in fin:
                        for ch in line:
                            if ch in stoi:
                                buf.append(stoi[ch])
                                if len(buf) >= FLUSH_TOKENS:
                                    np.array(buf, dtype=np.uint16).tofile(fout)
                                    token_count += len(buf)
                                    buf.clear()
            except Exception:
                continue

        if buf:
            np.array(buf, dtype=np.uint16).tofile(fout)
            token_count += len(buf)

    return out_path, token_count

def build_split(file_list, out_bin_path, split_name):
    print(f"[INFO] Building {split_name} with {NUM_WORKERS} workers...")

    chunks = [
        (i, file_list[i:i + CHUNK_SIZE])
        for i in range(0, len(file_list), CHUNK_SIZE)
    ]

    results = []
    total_tokens = 0

    with Pool(NUM_WORKERS) as pool:
        for out_path, tok in pool.imap_unordered(encode_worker, enumerate([c[1] for c in chunks])):
            results.append((out_path, tok))
            total_tokens += tok

    results.sort(key=lambda x: x[0])

    print(f"[INFO] Concatenating {len(results)} parts → {out_bin_path}")

    with open(out_bin_path, "wb") as fout:
        for part_path, _ in results:
            with open(part_path, "rb") as fin:
                fout.write(fin.read())

    for part_path, _ in results:
        os.remove(part_path)

    print(f"[OK] {split_name} tokens: {total_tokens:,}")
    return total_tokens


if __name__ == "__main__":
    train_files = load_file_list(TRAIN_LIST)
    val_files   = load_file_list(VAL_LIST)

    n_train = build_split(train_files, OUT_TRAIN_BIN, "train")
    n_val   = build_split(val_files, OUT_VAL_BIN, "val")

    meta = {
        "vocab_size": vocab_size,
        "itos": itos,
        "stoi": stoi,
        "desc": "Character-level ABC music dataset (parallel, nanoGPT compatible)",
    }
    with open(OUT_META, "wb") as f:
        pickle.dump(meta, f)

    print("[DONE] Saved train.bin, val.bin, meta.pkl")

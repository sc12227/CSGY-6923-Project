import os
import pickle
import numpy as np

BASE_DIR = os.path.dirname(__file__)
TEST_LIST = "../../../data/splits_unique/test.txt"
VOCAB_PATH = "../../../data/vocab_charlevel.txt"

OUT_TEST_BIN = os.path.join(BASE_DIR, "test.bin")
OUT_META = os.path.join(BASE_DIR, "meta.pkl")

chars = []
with open(VOCAB_PATH, "r", encoding="utf-8") as f:
    for line in f:
        tok = line.rstrip("\n")
        if tok == r"\n":
            tok = "\n"
        chars.append(tok)

chars = sorted(set(chars))
stoi = {ch: i for i, ch in enumerate(chars)}
vocab_size = len(chars)
print(f"[INFO] vocab_size = {vocab_size}")

def encode_list_to_bin(file_list_path, out_bin_path):
    buf = []
    total = 0
    with open(file_list_path, "r", encoding="utf-8") as flist, open(out_bin_path, "wb") as fout:
        for line in flist:
            path = line.strip()
            if not path or (not os.path.exists(path)):
                continue
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as fin:
                    for text_line in fin:
                        for ch in text_line:
                            idx = stoi.get(ch, None)
                            if idx is None:
                                continue
                            buf.append(idx)
                            if len(buf) >= 1_000_000:
                                np.array(buf, dtype=np.uint16).tofile(fout)
                                total += len(buf)
                                buf = []
            except Exception:
                continue

        if buf:
            np.array(buf, dtype=np.uint16).tofile(fout)
            total += len(buf)

    return total

if __name__ == "__main__":
    if not os.path.exists(OUT_META):
        raise RuntimeError(f"meta.pkl not found at {OUT_META}. Build train/val first.")

    print("[INFO] Building test.bin ...")
    n_test = encode_list_to_bin(TEST_LIST, OUT_TEST_BIN)
    print(f"[OK] test.bin tokens: {n_test:,}")
    print("[DONE] Saved test.bin under data/abc_char")

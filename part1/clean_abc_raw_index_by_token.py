# clean_abc_raw_index.py
# Clean ABC corpus BEFORE building 1B index
# DOES NOT COPY FILES — only writes an index of valid files.

import os
import argparse
import multiprocessing as mp
from functools import partial
from tqdm import tqdm

SRC_DIR = "../data/abc_raw"
OUT_INDEX = "../data/abc_clean_index.txt"

# Filtering criteria
MIN_TOKENS = 200
MAX_TOKENS = 100_000  # filter corrupted / extremely long

def count_tokens_charlevel(text):
    """Fast character-level token count."""
    return len(text)


def validate_file(path, min_tokens, max_tokens):
    """
    Validate file & return (is_valid, path)
    """
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()

        if len(text.strip()) == 0:
            return False, path  # empty file

        tok = count_tokens_charlevel(text)

        # filter by length
        if tok < min_tokens or tok > max_tokens:
            return False, path

        return True, path

    except Exception:
        return False, path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--min_tokens", type=int, default=MIN_TOKENS)
    parser.add_argument("--max_tokens", type=int, default=MAX_TOKENS)
    parser.add_argument("--out", type=str, default=OUT_INDEX)
    args = parser.parse_args()

    print("[INFO] Scanning input directory...")
    all_files = []
    for root, _, files in os.walk(SRC_DIR):
        for fn in files:
            if fn.endswith(".abc"):
                all_files.append(os.path.join(root, fn))

    print(f"[INFO] Total raw files: {len(all_files)}")

    worker = partial(
        validate_file,
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens
    )

    valid_files = []
    invalid = 0

    with mp.Pool(args.workers) as pool:
        for ok, path in tqdm(pool.imap_unordered(worker, all_files),
                             total=len(all_files),
                             desc="Validating ABC"):
            if ok:
                valid_files.append(path)
            else:
                invalid += 1

    # Write index file
    print(f"[INFO] Writing index to: {args.out}")
    with open(args.out, "w", encoding="utf-8") as f:
        for p in valid_files:
            f.write(p + "\n")

    print("\n===== CLEAN INDEX DONE =====")
    print("Total raw files  :", len(all_files))
    print("Valid (kept)     :", len(valid_files))
    print("Invalid (removed):", invalid)
    print("Index file saved :", args.out)


if __name__ == "__main__":
    main()

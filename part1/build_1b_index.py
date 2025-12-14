# build_1b_index_from_clean.py
# Build 1B-token index using the cleaned ABC index file

import os

CLEAN_INDEX = "../data/abc_clean_index.txt"
OUT_INDEX   = "../data/abc_1b_index.txt"

TARGET_TOKENS = 1_000_000_000   # 1B tokens


def count_tokens(path):
    """Character-level token count."""
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        return len(text)  # character-level tokens
    except:
        return 0


def main():
    if not os.path.exists(CLEAN_INDEX):
        print(f"[FATAL] Clean index not found: {CLEAN_INDEX}")
        return

    print("[INFO] Loading clean index...")
    with open(CLEAN_INDEX, "r", encoding="utf-8") as f:
        files = [line.strip() for line in f if line.strip()]

    print(f"[INFO] Total valid files listed: {len(files):,}")

    token_table = []
    total_clean_tokens = 0

    print("[INFO] Counting tokens for each file...")
    for path in files:
        tok = count_tokens(path)
        token_table.append((path, tok))
        total_clean_tokens += tok

    print(f"[INFO] Total tokens in clean corpus ≈ {total_clean_tokens:,}")

    # Build repeated 1B index
    repeated_index = []
    running = 0

    print("\n[INFO] Building repeated index until >= 1B tokens...")

    while running < TARGET_TOKENS:
        for path, tok in token_table:
            repeated_index.append(path)
            running += tok

            if running >= TARGET_TOKENS:
                break

        print(f"Progress: {running:,} tokens")

    # Save
    with open(OUT_INDEX, "w", encoding="utf-8") as f:
        for path in repeated_index:
            f.write(path + "\n")

    print("\n===== DONE =====")
    print(f"Target tokens: {TARGET_TOKENS:,}")
    print(f"Actual tokens: {running:,}")
    print(f"Index file saved: {OUT_INDEX}")
    print(f"Total index rows: {len(repeated_index):,}")


if __name__ == "__main__":
    main()

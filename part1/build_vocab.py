import os
import multiprocessing as mp
from tqdm import tqdm

ABC_DIR = "../data/abc_raw"
VOCAB_PATH = "../data/vocab_charlevel.txt"
NUM_WORKERS = 16


def extract_chars(path: str) -> set:
    local_vocab = set()
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as fin:
            for line in fin:
                local_vocab.update(line)
    except Exception:
        pass
    return local_vocab


def collect_abc_files(root_dir):
    files = []
    for root, _, fs in os.walk(root_dir):
        for f in fs:
            if f.endswith(".abc"):
                files.append(os.path.join(root, f))
    return files


def main():
    print("[INFO] Scanning ABC files...")
    abc_files = collect_abc_files(ABC_DIR)
    print(f"[INFO] Total files: {len(abc_files)}")

    global_vocab = set()

    with mp.Pool(NUM_WORKERS) as pool:
        for local_set in tqdm(
            pool.imap_unordered(extract_chars, abc_files, chunksize=50),
            total=len(abc_files),
            desc="Building char-level vocab"
        ):
            global_vocab |= local_set

    vocab = sorted(global_vocab)

    print("\n===== DONE =====")
    print("Final vocab size:", len(vocab))
    print("Saving vocab to:", VOCAB_PATH)

    with open(VOCAB_PATH, "w", encoding="utf-8") as f:
        for ch in vocab:
            if ch == "\n":
                f.write("\\n\n")
            else:
                f.write(ch + "\n")


if __name__ == "__main__":
    main()

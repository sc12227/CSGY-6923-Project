import os

CLEAN_INDEX = "../data/abc_clean_index.txt" 
OUT_DIR     = "../data/splits_unique"

TRAIN_RATIO = 0.98
VAL_RATIO   = 0.01
TEST_RATIO  = 0.01

MIN_TRAIN_TOKENS = 100_000_000
TARGET_TOTAL_TOKENS = 1_000_000_000  

os.makedirs(OUT_DIR, exist_ok=True)


def file_token_count(path: str) -> int:
    """Character-level token approximation."""
    try:
        return os.path.getsize(path)
    except:
        return 0


print("[INFO] Loading clean index...")
paths = []

with open(CLEAN_INDEX, "r", encoding="utf-8") as f:
    for line in f:
        p = line.strip()
        if p and os.path.exists(p):
            paths.append(p)

print(f"[INFO] Total clean valid files listed: {len(paths):,}")

print("[INFO] Counting tokens and selecting files up to 1B...")

paths_with_tok = []
total_tokens = 0

for p in paths:
    tok = file_token_count(p)
    if tok <= 0:
        continue

    if total_tokens + tok > TARGET_TOTAL_TOKENS:
        continue

    paths_with_tok.append((p, tok))
    total_tokens += tok

    if total_tokens >= TARGET_TOTAL_TOKENS:
        break

print(f"[INFO] Total tokens used for splitting ≈ {total_tokens:,}")

# Safety check
if len(paths_with_tok) == 0:
    raise RuntimeError("No files selected for splitting. Check index or token counts.")

assert abs((TRAIN_RATIO + VAL_RATIO + TEST_RATIO) - 1.0) < 1e-9

train_target = int(total_tokens * TRAIN_RATIO)
val_target   = int(total_tokens * VAL_RATIO)
test_target  = total_tokens - train_target - val_target

if train_target < MIN_TRAIN_TOKENS:
    raise RuntimeError(f"[FATAL] Train target < 100M tokens ({train_target:,}). Increase dataset.")

print("[INFO] Token targets:")
print(f"  Train target: {train_target:,}")
print(f"  Val target  : {val_target:,}")
print(f"  Test target : {test_target:,}")

train_list, val_list, test_list = [], [], []
train_tok = val_tok = test_tok = 0

split = "train"

for p, tok in paths_with_tok:

    if split == "train":
        train_list.append(p)
        train_tok += tok
        if train_tok >= train_target:
            split = "val"

    elif split == "val":
        val_list.append(p)
        val_tok += tok
        if val_tok >= val_target:
            split = "test"

    else:
        test_list.append(p)
        test_tok += tok

train_set = set(train_list)
val_set   = set(val_list)
test_set  = set(test_list)

if (train_set & val_set) or (train_set & test_set) or (val_set & test_set):
    raise RuntimeError("[FATAL] Split overlap detected!")

def write_list(path, items):
    with open(path, "w", encoding="utf-8") as f:
        for x in items:
            f.write(x + "\n")

write_list(os.path.join(OUT_DIR, "train.txt"), train_list)
write_list(os.path.join(OUT_DIR, "val.txt"),   val_list)
write_list(os.path.join(OUT_DIR, "test.txt"),  test_list)

print("\n===== SPLIT DONE (DISJOINT) =====")
print(f"Train: files={len(train_list):,}, tokens={train_tok:,}")
print(f"Val  : files={len(val_list):,}, tokens={val_tok:,}")
print(f"Test : files={len(test_list):,}, tokens={test_tok:,}")
print("[OK] No overlap between splits.")
print(f"[OK] Saved to: {OUT_DIR}")

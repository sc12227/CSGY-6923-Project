# clean_abc_delete.py
# Directly delete ABC files based on length thresholds
# No copying, no new directory.

import os

SRC = "../data/abc_raw"

MAX_LEN = 500_000     # recommended: remove extremely long corrupted files
MIN_LEN = 200         # recommended: remove too-short meaningless files

count = 0
removed = 0
kept = 0

for root, _, files in os.walk(SRC):
    for name in files:
        if not name.endswith(".abc"):
            continue

        path = os.path.join(root, name)
        
        try:
            size = os.path.getsize(path)
        except:
            continue
        
        count += 1
        
        # Delete file if too short or too long
        if size < MIN_LEN or size > MAX_LEN:
            try:
                os.remove(path)
                removed += 1
            except Exception as e:
                print(f"[ERROR] Failed to delete {path}: {e}")
            continue

        kept += 1

print("===== CLEANING DONE =====")
print(f"Total files scanned : {count}")
print(f"Kept valid files    : {kept}")
print(f"Deleted invalid     : {removed}")

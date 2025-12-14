import os
import glob
import argparse
import subprocess
import multiprocessing as mp
from functools import partial
from tqdm import tqdm

RAW_MIDI_DIR = "../data/midi_raw"
ABC_DIR      = "../data/abc_raw"
FAILED_LOG   = "../data/failed_midi.txt"

MIDI2ABC_BIN = "/root/miniconda3/lib/python3.12/site-packages/symusic/bin/midi2abc"

os.makedirs(ABC_DIR, exist_ok=True)


def convert_single(midi_path: str, min_abc_len: int = 10) -> bool:
    try:
        base = os.path.splitext(os.path.basename(midi_path))[0]
        out_path = os.path.join(ABC_DIR, base + ".abc")

        if os.path.exists(out_path) and os.path.getsize(out_path) > min_abc_len:
            return True

        if not os.path.exists(MIDI2ABC_BIN):
            return False

        result = subprocess.run(
            [MIDI2ABC_BIN, midi_path, "-o", out_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=False
        )

        if result.returncode != 0:
            return False

        if (not os.path.exists(out_path)) or (os.path.getsize(out_path) <= min_abc_len):
            return False

        return True

    except:
        return False


def run_parallel(midi_files, workers: int, min_abc_len: int = 10):
    convert = partial(convert_single, min_abc_len=min_abc_len)
    success_flags = []

    with mp.Pool(workers) as pool:
        for result in tqdm(
            pool.imap_unordered(convert, midi_files),
            total=len(midi_files),
            desc=f"Converting MIDI → ABC ({workers} workers)"
        ):
            success_flags.append(result)

    return sum(success_flags), len(success_flags) - sum(success_flags)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--min_abc_len", type=int, default=10)
    args = parser.parse_args()

    midi_files = sorted(set(
        glob.glob(os.path.join(RAW_MIDI_DIR, "**/*.mid"), recursive=True) +
        glob.glob(os.path.join(RAW_MIDI_DIR, "**/*.MID"), recursive=True)
    ))

    print(f"[INFO] MIDI files: {len(midi_files)}")
    print(f"[INFO] Writing ABC to: {ABC_DIR}")

    success, fail = run_parallel(midi_files, args.workers, args.min_abc_len)

    print("\n==== DONE ====")
    print(f"Success: {success}")
    print(f"Failed : {fail}")

    failed_files = []
    for m in midi_files:
        base = os.path.splitext(os.path.basename(m))[0]
        out_path = os.path.join(ABC_DIR, base + ".abc")

        if (not os.path.exists(out_path)) or (os.path.getsize(out_path) <= args.min_abc_len):
            failed_files.append(m)

    if failed_files:
        with open(FAILED_LOG, "w") as f:
            for p in failed_files:
                f.write(p + "\n")
        print(f"[INFO] Logged {len(failed_files)} failures to {FAILED_LOG}")
    else:
        print("[INFO] All ABC files valid.")


if __name__ == "__main__":
    main()

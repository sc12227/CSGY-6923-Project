import os
import random
import argparse
import subprocess
import tempfile


def extract_abc(text):
    lines = text.splitlines()
    cleaned = []
    for ln in lines:
        if ln.strip().startswith("="):
            continue
        cleaned.append(ln)
    return "\n".join(cleaned).strip()


def read_random_prefixes(test_list_path, k=10, seed=123):
    random.seed(seed)

    paths = []
    with open(test_list_path, "r", encoding="utf-8") as f:
        for line in f:
            p = line.strip()
            if p:
                paths.append(p)

    random.shuffle(paths)
    prefixes = []

    for p in paths:
        try:
            with open(p, "r", encoding="utf-8", errors="ignore") as fin:
                lines = fin.read().splitlines()

            header = []
            body = []

            for ln in lines:
                if ":" in ln[:2]:
                    header.append(ln)
                else:
                    body.append(ln)

                if len(header) >= 4 and len(body) >= 5:
                    break

            if len(header) >= 2 and len(body) >= 3:
                prefix = "\n".join(header + body)
                prefixes.append(prefix)

            if len(prefixes) >= k:
                break

        except Exception:
            continue

    if not prefixes:
        raise RuntimeError("Could not extract valid ABC prefixes.")

    return prefixes


def run_sample(out_dir, start_text, max_new_tokens, temperature, top_k, seed):
    cmd = [
        "python", "sample.py",
        f"--out_dir={out_dir}",
        "--num_samples=1",
        f"--max_new_tokens={max_new_tokens}",
        f"--temperature={temperature}",
        f"--top_k={top_k}",
        f"--seed={seed}",
    ]

    if start_text is not None:
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".abc") as f:
            f.write(start_text)
            start_path = f.name
        cmd.append(f"--start=FILE:{start_path}")

    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(res.stderr[-2000:])

    return extract_abc(res.stdout)




def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", default="./part4_results/epoch_03")
    ap.add_argument("--out_root", default="./part4_results/samples")
    ap.add_argument("--test_list", default="../../data/splits_unique/test.txt")
    ap.add_argument("--n_uncond", type=int, default=10)
    ap.add_argument("--n_cond", type=int, default=10)
    ap.add_argument("--max_new_tokens", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    if not os.path.exists(os.path.join(args.ckpt_dir, "ckpt.pt")):
        raise FileNotFoundError(f"ckpt.pt not found in {args.ckpt_dir}")

    os.makedirs(args.out_root, exist_ok=True)
    uncond_dir = os.path.join(args.out_root, "unconditional_abc")
    cond_dir = os.path.join(args.out_root, "conditional_abc")
    os.makedirs(uncond_dir, exist_ok=True)
    os.makedirs(cond_dir, exist_ok=True)

    temps = [0.8, 0.9, 1.0, 1.1]
    topks = [40, 50, 80]

    for i in range(args.n_uncond):
        txt = run_sample(
            out_dir=args.ckpt_dir,
            start_text=None,
            max_new_tokens=args.max_new_tokens,
            temperature=temps[i % len(temps)],
            top_k=topks[i % len(topks)],
            seed=args.seed + i * 17,
        )

        out_path = os.path.join(
            uncond_dir,
            f"uncond_{i:02d}.abc"
        )
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(txt)

    prefixes = read_random_prefixes(
        args.test_list, k=args.n_cond, seed=args.seed
    )

    for i, pref in enumerate(prefixes):
        txt = run_sample(
            out_dir=args.ckpt_dir,
            start_text=pref,
            max_new_tokens=args.max_new_tokens,
            temperature=temps[(i + 1) % len(temps)],
            top_k=topks[(i + 1) % len(topks)],
            seed=args.seed + 1000 + i * 19,
        )

        out_path = os.path.join(
            cond_dir,
            f"cond_{i:02d}.abc"
        )
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(txt)

    print("[DONE] Samples saved to:")
    print(" ", uncond_dir)
    print(" ", cond_dir)


if __name__ == "__main__":
    main()

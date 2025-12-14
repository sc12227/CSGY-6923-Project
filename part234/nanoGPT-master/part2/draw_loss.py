# plot_training_curves.py
import re
import matplotlib.pyplot as plt
from pathlib import Path

LOG_DIR = Path("scaling_logs")  # 改成你的 log 目录
MODELS = ["tiny", "small", "medium", "large"]

def parse_log(log_path):
    """
    从 nanoGPT log 中提取 (iter, loss)
    """
    iters = []
    losses = []

    pattern = re.compile(r"iter\s+(\d+):\s+loss\s+([0-9.]+)")

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                iters.append(int(m.group(1)))
                losses.append(float(m.group(2)))

    return iters, losses


def main():
    plt.figure(figsize=(8, 5))

    for name in MODELS:
        log_path = LOG_DIR / f"{name}.log"
        if not log_path.exists():
            print(f"[WARN] {log_path} not found, skipping.")
            continue

        iters, losses = parse_log(log_path)

        if not iters:
            print(f"[WARN] No valid iter/loss found in {name}.log")
            continue

        plt.plot(iters, losses, label=name.capitalize())

        print(
            f"[OK] {name}: points={len(iters)}, "
            f"first_loss={losses[0]:.3f}, last_loss={losses[-1]:.3f}"
        )

    plt.xlabel("Iteration")
    plt.ylabel("Training Loss")
    plt.title("Training Loss Curves (1 Epoch)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig("training_curves.png", dpi=200)
    plt.show()


if __name__ == "__main__":
    main()

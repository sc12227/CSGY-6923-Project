import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Your experimental results
# -----------------------------
models = ["Tiny", "Small", "Medium", "Large", "XL"]
params_m = np.array([0.41, 3.17, 10.66, 25.23, 59.06])   # Params in millions
time_epoch_min = np.array([15.44, 23.29, 34.24, 51.61, 91.47])  # minutes per epoch

# -----------------------------
# Plot
# -----------------------------
plt.figure(figsize=(7, 5))
plt.plot(params_m, time_epoch_min, marker="o")

plt.xscale("log")
plt.xlabel("Model Size (Million Parameters, log scale)")
plt.ylabel("Wall-clock Time per Epoch (minutes)")
plt.title("Compute Cost vs Model Size (ABC Char Transformer)")

plt.grid(True, which="both", linestyle="--", alpha=0.4)

# Save
plt.tight_layout()
plt.savefig("compute_cost_vs_model_size.png", dpi=200)
plt.show()

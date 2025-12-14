import numpy as np
import matplotlib.pyplot as plt

models = ["Tiny", "Small", "Medium", "Large", "XL"]
params_m = np.array([0.41, 3.17, 10.66, 25.23, 59.06])

# Rule of thumb: ~12 MB per 1M params (FP16 training)
gpu_mem_gb = params_m * 12 / 1024  # convert MB to GB

plt.figure(figsize=(7, 5))
plt.plot(params_m, gpu_mem_gb, marker="o", color="orange")

plt.xscale("log")
plt.xlabel("Model Size (Million Parameters, log scale)")
plt.ylabel("Estimated GPU Memory Usage (GB)")
plt.title("Estimated GPU Memory Usage vs Model Size")

plt.grid(True, which="both", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("gpu_memory_estimate_vs_model_size.png", dpi=200)
plt.show()

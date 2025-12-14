import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

transformer = {
    "name": "Transformer",
    "params_m": np.array([0.41, 3.17, 10.66, 25.23, 59.06]),
    "val_loss": np.array([0.6377, 0.3987, 0.3502, 0.3349, 0.3211]),
    "time_epoch": np.array([15.61, 23.29, 34.24, 51.61, 91.47]),
    "gpu_mem": np.array([0.005, 0.038, 0.13, 0.30, 0.71]),
}

rnn = {
    "name": "RNN (LSTM)",
    "params_m": np.array([1.10, 6.41, 19.05, 42.19]),
    "val_loss": np.array([0.5775, 0.5098, 0.4799, 0.5317]),
    "time_epoch": np.array([10.44, 13.70, 32.27, 63.91]),
    "gpu_mem": np.array([0.05, 0.18, 0.48, 0.95]),
}

def scaling_law(N, a, alpha, c):
    return a * (N ** (-alpha)) + c

def fit_scaling(params_m, losses):
    popt, _ = curve_fit(
        scaling_law,
        params_m,
        losses,
        maxfev=10000
    )
    return popt
t_a, t_alpha, t_c = fit_scaling(transformer["params_m"], transformer["val_loss"])
r_a, r_alpha, r_c = fit_scaling(rnn["params_m"], rnn["val_loss"])

print("===== Scaling Exponents =====")
print(f"Transformer alpha = {t_alpha:.4f}")
print(f"RNN alpha         = {r_alpha:.4f}")

plt.figure(figsize=(6, 4))
plt.scatter(rnn["params_m"], rnn["val_loss"], label="RNN (LSTM)", s=60)

x = np.logspace(np.log10(rnn["params_m"].min()),
                np.log10(rnn["params_m"].max()), 200)
plt.plot(x, scaling_law(x, r_a, r_alpha, r_c), linestyle="--")

plt.xscale("log")
plt.xlabel("Parameters (Millions)")
plt.ylabel("Validation Loss (1 epoch)")
plt.title("RNN Scaling Law")
plt.legend()
plt.tight_layout()
plt.savefig("rnn_scaling.png", dpi=200)

plt.figure(figsize=(6, 4))
plt.scatter(transformer["params_m"], transformer["val_loss"],
            label="Transformer", s=60)

x = np.logspace(np.log10(transformer["params_m"].min()),
                np.log10(transformer["params_m"].max()), 200)
plt.plot(x, scaling_law(x, t_a, t_alpha, t_c), linestyle="--")

plt.xscale("log")
plt.xlabel("Parameters (Millions)")
plt.ylabel("Validation Loss (1 epoch)")
plt.title("Transformer Scaling Law")
plt.legend()
plt.tight_layout()
plt.savefig("transformer_scaling.png", dpi=200)

plt.figure(figsize=(7, 5))

plt.scatter(transformer["params_m"], transformer["val_loss"],
            label=f"Transformer (α={t_alpha:.2f})", s=60)
plt.scatter(rnn["params_m"], rnn["val_loss"],
            label=f"RNN (α={r_alpha:.2f})", s=60)

x_t = np.logspace(np.log10(transformer["params_m"].min()),
                  np.log10(transformer["params_m"].max()), 200)
x_r = np.logspace(np.log10(rnn["params_m"].min()),
                  np.log10(rnn["params_m"].max()), 200)

plt.plot(x_t, scaling_law(x_t, t_a, t_alpha, t_c), linestyle="--")
plt.plot(x_r, scaling_law(x_r, r_a, r_alpha, r_c), linestyle="--")

plt.xscale("log")
plt.xlabel("Parameters (Millions)")
plt.ylabel("Validation Loss (1 epoch)")
plt.title("Transformer vs RNN Scaling Comparison")
plt.legend()
plt.tight_layout()
plt.savefig("rnn_vs_transformer_scaling.png", dpi=200)

print("\n===== Efficiency Analysis =====")

print("\nSample Efficiency:")
print("Transformer achieves lower validation loss at comparable or larger parameter counts.")
print("RNN shows diminishing returns and degradation at largest scale.")

print("\nCompute Efficiency (Time per Epoch):")
for i in range(len(transformer["params_m"])):
    print(f"Transformer {transformer['params_m'][i]:.2f}M: "
          f"{transformer['time_epoch'][i]:.2f} min")

for i in range(len(rnn["params_m"])):
    print(f"RNN {rnn['params_m'][i]:.2f}M: "
          f"{rnn['time_epoch'][i]:.2f} min")

print("\nMemory Efficiency (GB per 10M params):")
print("Transformer:")
for p, m in zip(transformer["params_m"], transformer["gpu_mem"]):
    print(f"  {p:.2f}M → {m / p * 10:.2f} GB / 10M params")

print("RNN:")
for p, m in zip(rnn["params_m"], rnn["gpu_mem"]):
    print(f"  {p:.2f}M → {m / p * 10:.2f} GB / 10M params")

print("\nConclusion:")
if t_alpha > r_alpha:
    print("Transformer scales better than RNN (higher alpha).")
else:
    print("RNN scales better (unexpected).")

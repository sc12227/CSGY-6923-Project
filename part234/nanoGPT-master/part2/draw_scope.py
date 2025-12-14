import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# 数据（单位：M parameters）
params = np.array([0.41, 3.17, 10.66, 25.23, 59.06])
val_loss = np.array([0.6377, 0.3987, 0.3502, 0.3349, 0.3211])

def scaling_law(N, a, alpha, c):
    return a * (N ** (-alpha)) + c

popt, _ = curve_fit(scaling_law, params, val_loss)
a, alpha, c = popt

print(f"Fitted alpha = {alpha:.4f}")

x = np.linspace(params.min(), params.max(), 300)
y = scaling_law(x, a, alpha, c)

plt.figure(figsize=(6,4))
plt.scatter(params, val_loss, label="Models")
plt.plot(x, y, label=f"Fit: alpha={alpha:.3f}")
plt.xscale("log")
plt.xlabel("Model size (M parameters)")
plt.ylabel("Validation loss after 1 epoch")
plt.title("Transformer Scaling on ABC Music (char-level)")
plt.legend()
plt.tight_layout()
plt.savefig("scaling_plot_transformer.png")
plt.show()

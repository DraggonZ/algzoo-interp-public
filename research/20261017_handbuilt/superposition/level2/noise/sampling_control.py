"""Control at equal cost: the decision-level quantities run each copy on 720 canonical lists. How well does plain
sampling of 720 (and 2,160 = three value settings' worth; 10,000) random lists predict each held-out copy's accuracy?"""
import numpy as np, torch as th
from noiselib import net, accuracy, DEV
W = net("ARC"); T = np.load("../truth/noisy_truth.npz")
for s in ("0.001", "0.0025", "0.005", "0.01"):
    y = T[f"stats_{s}"][:, 0]; line = f"s = {s}:"
    for n in (720, 2160, 10000):
        e = np.array([accuracy(W, th.as_tensor(T[f"dw_{s}"], device=DEV), n_lists=n, seed=1000 + r, chunk=n) - y for r in range(5)])
        line += f"  {n} random lists RMSE {100 * np.sqrt(np.mean(e ** 2)):.2f} points;"
    print(line)

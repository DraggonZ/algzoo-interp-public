"""Readout-only ablation: replace one group's final activity by its mean in the output (dynamics untouched).
Accuracy on lists whose second-largest is a revealed old largest versus a kept number."""
import numpy as np
from algzoo.core import example

wi, wh, wo = (np.asarray(w, dtype=float) for w in example()); wi = wi.reshape(-1)
groups = {"none": [], "store, detector (2,4)": [2, 4], "neuron 2": [2], "neuron 4": [4], "gate (5)": [5],
          "neuron 3": [3], "cancellers (1,6,7,8)": [1, 6, 7, 8]}
rng = np.random.default_rng(24)
X = rng.standard_normal((1_000_000, 10)); B, n = X.shape
h = np.zeros((B, 16))
for t in range(n): h = np.maximum(h @ wh.T + X[:, t:t + 1] * wi, 0)
z = np.zeros((B, 2)); m1p = np.stack([np.max(np.c_[z, X[:, :t]], 1) for t in range(n)], 1)
record = X > m1p; o = X.argsort(1); sec, lar = o[:, -2], o[:, -1]; r = np.arange(B)
rev = record[r, sec] & (sec < lar); kept = ~record[r, sec] & (X[r, sec] > 0)
mu = h.mean(0)
for name, G in groups.items():
    hh = h.copy(); hh[:, G] = mu[G]
    P = (hh @ wo.T).argmax(1)
    print(f"{name:22s} revealed-second lists {(P[rev] == sec[rev]).mean():.4f}   kept-second lists {(P[kept] == sec[kept]).mean():.4f}")

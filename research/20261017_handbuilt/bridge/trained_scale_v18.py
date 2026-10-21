"""v18 built at the trained network's scale (gains 10, vote 10 per unit: build_v18(g=10, a=10, w=10, lam=10, c=-9,
eps=0.1), as in robust_twin/sensitivity.py). Accuracy, then save its weights for the matched heatmap."""
import numpy as np
from algzoo.handbuilt_v18 import build_v18
rng = np.random.default_rng(0); X = rng.standard_normal((300000, 10)); y = np.argsort(X, 1)[:, -2]
P = [np.asarray(p, float) for p in build_v18(g=10, a=10, w=10, lam=10.0, c=-9.0, eps=0.1).params]
h = np.zeros((len(X), 16))
for t in range(10): h = np.maximum(h @ P[1].T + X[:, t:t + 1] * P[0], 0)
print("v18 at trained scale: accuracy", ((h @ P[2].T).argmax(1) == y).mean(), " largest |weight|", max(np.abs(p).max() for p in P))
np.savez("v18_trained_scale.npz", wi=P[0], wh=P[1], wo=P[2])

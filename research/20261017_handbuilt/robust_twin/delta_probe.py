"""v19 with the wipe threshold below the largest (delta), trained scale, kappa 0 and 0.5: accuracy, accuracy on lists
whose second is an overtaken record with a small lead (under 5%), and noise tolerance. delta 0 is the earlier v19."""
import numpy as np
from algzoo.handbuilt_v19 import build_v19
import simlib

X = simlib.lists(81, 100_000); y = X.argsort(1)[:, -2]; o = X.argsort(1)[:, ::-1]; rng = np.random.default_rng(0)
m1 = np.take_along_axis(X, o[:, :1], 1)[:, 0]; m2 = np.take_along_axis(X, o[:, 1:2], 1)[:, 0]
small = (o[:, 1] < o[:, 0]) & ((m1 - m2) / np.maximum(m1, 1e-9) < 0.05)
near_after = (o[:, 1] > o[:, 0]) & ((m1 - m2) / np.maximum(m1, 1e-9) < 0.1)
for kappa in (0.0, 0.5):
    for delta in (0.0, 0.02, 0.05, 0.1, 0.2):
        P = tuple(np.asarray(p, float) for p in build_v19(g=10, a=10, w=10, lam=10.0, c=-9.0, eps=0.1, kappa=kappa, delta=delta).params)
        L, _, H = simlib.run(X, params=P); a = L.argmax(1)
        typ = np.array([np.median(H[:, :, n][H[:, :, n] > 0]) if (H[:, :, n] > 0).any() else 1.0 for n in range(16)])
        h = np.zeros((len(X), 16))
        for t in range(10):
            h = np.maximum(h @ P[1].T + X[:, t:t + 1] * P[0], 0); h = np.maximum(h + 0.01 * typ * rng.standard_normal(h.shape) * (h > 0), 0)
        print(f"kappa {kappa} delta {delta:4.2f}: accuracy {np.mean(a == y):.4f}; small-lead overtaken record {np.mean(a[small] == y[small]):.3f}; second after the largest within 10% {np.mean(a[near_after] == y[near_after]):.3f}; 1% noise {np.mean((h @ P[2].T).argmax(1) == y):.3f}", flush=True)

"""Idealized probe of the hybrid wipe (input wipe plus a cleanup one step later): run v19 (no flip, beta 0) with the
timer set to exactly zero at every record step before the last; at the last step only the input wipe acts. Timer levels
g = 10, 5, 3; kappa 0, 0.5, 1. Accuracy, and accuracy under noise of 1e-3 and 1e-2 x typical values (the idealized wipe
applied in the noisy runs too). An upper bound on what building the hybrid can give, not a built network."""
import numpy as np
from algzoo.handbuilt_v19 import build_v19
import simlib

X = simlib.lists(78, 50_000); y = X.argsort(1)[:, -2]; T = [3, 4, 5, 6]; rng = np.random.default_rng(0)
cm = np.maximum.accumulate(X, 1); prev = np.c_[np.zeros((len(X), 1)), cm[:, :-1]]; rec = X > np.maximum(prev, 0)
def run(P, s, typ):
    h = np.zeros((len(X), 16))
    for t in range(10):
        h = np.maximum(h @ P[1].T + X[:, t:t + 1] * P[0], 0)
        if t < 9:
            h[np.ix_(rec[:, t], T)] = 0.0
        if s:
            h = np.maximum(h + s * typ * rng.standard_normal(h.shape) * (h > 0), 0)
    return ((h @ P[2].T).argmax(1) == y).mean()
for g in (10, 5, 3):
    for kappa in (0.0, 0.5, 1.0):
        P = tuple(np.asarray(p, float) for p in build_v19(beta=0.0, g=g, a=10, w=10, lam=10.0, c=-9.0, eps=0.1, kappa=kappa).params)
        H = simlib.run(X, params=P)[2]; typ = np.array([np.median(H[:, :, n][H[:, :, n] > 0]) if (H[:, :, n] > 0).any() else 1.0 for n in range(16)])
        print(f"g {g:2d} kappa {kappa}: accuracy {run(P, 0, typ):.4f}; noise 1e-3 {run(P, 1e-3, typ):.4f}; 1e-2 {run(P, 1e-2, typ):.4f}", flush=True)

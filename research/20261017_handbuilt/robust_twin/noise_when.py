"""v19 at trained scale (kappa 0.5, delta 0.02): does 1% noise on the timer neurons (clock and flip) hurt mostly
through accumulation over steps, or at the final step? Noise on the timer only at the last step, at the last 3 steps,
at every step; and on all neurons at every step. The trained network's cancelling neurons for comparison."""
import numpy as np
from algzoo.handbuilt_v19 import build_v19
import simlib
X = simlib.lists(86, 50_000); y = X.argsort(1)[:, -2]; rng = np.random.default_rng(0)
def run(P, typ, ns, steps, s=0.01):
    h = np.zeros((len(X), 16))
    for t in range(10):
        h = np.maximum(h @ P[1].T + X[:, t:t + 1] * P[0], 0)
        if t in steps:
            e = np.zeros_like(h); e[:, ns] = s * typ[ns] * rng.standard_normal((len(X), len(ns)))
            h = np.maximum(h + e * (h > 0), 0)
    return np.mean((h @ P[2].T).argmax(1) == y)
for name, P, T in (("v19 kappa 0.5 delta 0.02", tuple(np.asarray(p, float) for p in build_v19(g=10, a=10, w=10, lam=10.0, c=-9.0, eps=0.1, kappa=0.5, delta=0.02).params), [3, 4, 5, 6]),
                   ("trained", (simlib.WI, simlib.WH, simlib.WO), [1, 6, 7, 8])):
    H = simlib.run(X, params=P)[2]; typ = np.array([np.median(H[:, :, n][H[:, :, n] > 0]) if (H[:, :, n] > 0).any() else 1.0 for n in range(16)])
    print(f"{name}: no noise {run(P, typ, T, [], 0):.3f}; timer, last step {run(P, typ, T, [9]):.3f}; timer, last 3 steps {run(P, typ, T, [7, 8, 9]):.3f}; timer, every step {run(P, typ, T, range(10)):.3f}; all neurons, every step {run(P, typ, list(range(16)), range(10)):.3f}", flush=True)

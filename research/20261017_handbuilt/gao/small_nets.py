"""Do ARC's fully understood small networks of the same family compute by cancellation too? For M2,2 and M4,3 (zoo
models, several seeds) and ARC's M16,10: accuracy, median lead of the winner over the runner-up, its ratio to the pushes
on the lead, and accuracy with noise of 0.3%, 1% and 3% of each neuron's typical value. 100,000 lists."""
import numpy as np
from algzoo.core import zoo, example
import simlib


def stats(name, P, n):
    X = np.random.default_rng(2020).standard_normal((100_000, n)); y = X.argsort(1)[:, -2]; B = len(X); r = np.arange(B)
    wi, wh, wo = P; d = len(wi); h = np.zeros((B, d)); Hs = []
    for t in range(n):
        h = np.maximum(h @ wh.T + X[:, t:t + 1] * wi, 0); Hs.append(h)
    Hs = np.stack(Hs, 1); L = h @ wo.T; k = L.argmax(1); j = np.argsort(L, 1)[:, -2]
    lead = L[r, k] - L[r, j]; push = np.abs((wo[k] - wo[j]) * h).sum(1)
    typ = np.array([np.median(Hs[:, :, q][Hs[:, :, q] > 0]) if (Hs[:, :, q] > 0).any() else 1.0 for q in range(d)])
    rng = np.random.default_rng(0); acc = []
    for s in (0.003, 0.01, 0.03):
        h = np.zeros((B, d))
        for t in range(n):
            h = np.maximum(h @ wh.T + X[:, t:t + 1] * wi, 0); h = np.maximum(h + s * typ * rng.standard_normal(h.shape) * (h > 0), 0)
        acc.append(((h @ wo.T).argmax(1) == y).mean())
    print(f"{name:12s}: accuracy {(k == y).mean():.4f}; median lead {np.median(lead):.2f}; lead / pushes {np.median(lead / np.maximum(push, 1e-12)):.3f}; "
          f"with noise 0.3% {acc[0]:.4f}, 1% {acc[1]:.4f}, 3% {acc[2]:.4f}", flush=True)


for (d, n) in ((2, 2), (4, 3)):
    for seed in (None, 0, 1, 2):
        try:
            P = tuple(np.asarray(p, float) for p in (zoo(d, n) if seed is None else zoo(d, n, seed))); P = (P[0].reshape(-1), P[1], P[2])
            stats(f"M{d},{n} " + ("ARC" if seed is None else f"seed {seed}"), P, n)
        except Exception as e:
            print(f"M{d},{n} seed {seed}: {e}")
stats("M16,10 ARC", (simlib.WI, simlib.WH, simlib.WO), 10)

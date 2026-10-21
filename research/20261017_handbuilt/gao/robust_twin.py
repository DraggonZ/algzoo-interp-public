"""Can the twin be built with the trained network's margins? Its fragility comes from large levels that keep signed
signals linear through the ReLUs (timer level g (m1 - x), memory level lam m1, gate gains w and a), which the readout
cancels exactly. Builds: one principled build with small levels (g = 3, lam = 1, w = a = 30; chosen from the sizes involved,
not searched); then the trained network's own measured gains (timer level, detector and store about 10-16 x;
memory level 10 x the largest, as the twin already has), with and without the tie-breaker; then also the vote gain (1 - c) at the trained network's scale of about 10: then the tie-breaker raised to 1% of a vote (the twin decides about 17% of lists by an exact tie between
the largest's and the second-largest's positions, broken by eps m1): accuracy, margin against the pushes on the winner's lead, and accuracy with noise on true values."""
import numpy as np
import simlib
from algzoo.handbuilt_v18 import build_v18

X = simlib.lists(1111, 100_000); y = X.argsort(1)[:, -2]; rng = np.random.default_rng(0)
for label, kw in (("twin (g 1e4, lam 10, w 1e4, a 1000)", {}), ("twin gains 30", dict(g=30, w=30, a=20)),
                  ("small levels (g 3, lam 1, w 30, a 30)", dict(g=3, lam=1, w=30, a=30)),
                  ("trained network's gains (g = a = w = 10)", dict(g=10, a=10, w=10)),
                  ("trained gains, no tie-breaker (eps 0)", dict(g=10, a=10, w=10, eps=0.0)),
                  ("trained gains, memory level 1 (lam 1)", dict(g=10, a=10, w=10, lam=1.0)),
                  ("all at trained scale: vote gain 10 (c -9)", dict(g=10, a=10, w=10, lam=10.0, c=-9.0)),
                  ("trained scale, tie-breaker 1% of a vote", dict(g=10, a=10, w=10, lam=10.0, c=-9.0, eps=0.1)),
                  ("original twin, tie-breaker 1% of a vote", dict(eps=0.0025)),
                  ("trained scale, tie-breaker 10% of a vote", dict(g=10, a=10, w=10, lam=10.0, c=-9.0, eps=1.0)),
                  ("trained scale, tie-breaker 50% of a vote", dict(g=10, a=10, w=10, lam=10.0, c=-9.0, eps=5.0))):
    P = tuple(np.asarray(p, float) for p in build_v18(**kw).params)
    L, _, H = simlib.run(X, params=P); k = L.argmax(1); j = np.argsort(L, 1)[:, -2]
    lead = L[np.arange(len(L)), k] - L[np.arange(len(L)), j]; push = np.abs((P[2][k] - P[2][j]) * H[:, 9]).sum(1)
    typ = np.array([np.median(H[:, :, n][H[:, :, n] > 0]) if (H[:, :, n] > 0).any() else 1.0 for n in range(16)])
    tol = []
    for s in (1e-4, 1e-3, 1e-2):
        h = np.zeros((len(X), 16))
        for t in range(10):
            h = np.maximum(h @ P[1].T + X[:, t:t + 1] * P[0], 0); h = np.maximum(h + s * typ * rng.standard_normal(h.shape) * (h > 0), 0)
        tol.append(f"{s:g}: {((h @ P[2].T).argmax(1) == y).mean():.3f}")
    print(f"{label:38s}: accuracy {(k == y).mean():.3f}; lead / pushes median {np.median(lead / np.maximum(push, 1e-12)):.2e}; with noise " + ", ".join(tol), flush=True)

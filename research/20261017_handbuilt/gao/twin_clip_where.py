"""At a low timer level (g = 3, trained-scale twin, 83.5%), whose clipping costs accuracy? Run the twin with the ReLU
removed (identity) for one group at a time: clock (4-6), flip (3), memory (8-15), keep gate (2), flag (7)."""
import numpy as np
from algzoo.handbuilt_v18 import build_v18
import simlib

X = simlib.lists(61, 50_000); y = X.argsort(1)[:, -2]
groups = {"none": [], "clock": [4, 5, 6], "flip": [3], "clock and flip": [3, 4, 5, 6], "memory": list(range(8, 16)), "keep gate": [2], "flag": [7]}
for g in (10, 3, 1):
    P = tuple(np.asarray(p, float) for p in build_v18(g=g, a=10, w=10, lam=10.0, c=-9.0, eps=0.1).params)
    out = []
    for lab, ns in groups.items():
        lin = np.zeros(16, bool); lin[ns] = True; h = np.zeros((len(X), 16))
        for t in range(10):
            z = h @ P[1].T + X[:, t:t + 1] * P[0]; h = np.where(lin, z, np.maximum(z, 0))
        out.append(f"{lab} {((h @ P[2].T).argmax(1) == y).mean():.3f}")
    print(f"g {g:3g}, accuracy with no clipping in: " + "; ".join(out), flush=True)

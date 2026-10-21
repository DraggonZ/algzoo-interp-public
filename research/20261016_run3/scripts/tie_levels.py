"""Does a placement's second-third tie margin carry over between levels? Margin M0 = score[p2] - score[p3] at m2 = m3,
for (m2, m1 - m2) in a small grid; correlation and sign agreement across the 720 placements between grid points."""
import itertools
import sys

import numpy as np

sys.path.insert(0, "/data/output/research/algzoo/20261016_run3/scripts")
from band_origin import T, b, scores  # noqa: E402

from algzoo.core import example, zoo  # noqa: E402

n = len(T); ar = np.arange(n)
grid = [(0.9, 0.2), (0.9, 0.5), (0.9, 1.0), (0.5, 0.5), (1.5, 0.5)]
for name in sys.argv[1:] or ["example", 2, 3, 0, 4, 1]:
    p = example() if name == "example" else zoo(16, 10, int(name))
    M = {}
    for m2, g in grid:
        X = np.full((n, 10), b); X[ar, T[:, 0]] = m2 + g; X[ar, T[:, 1]] = m2; X[ar, T[:, 2]] = m2
        S = scores(p, X); M[(m2, g)] = S[ar, T[:, 1]] - S[ar, T[:, 2]]
    ref = M[(0.9, 0.5)]
    out = []
    for k in grid:
        if k == (0.9, 0.5):
            continue
        out.append(f"m2 {k[0]}, gap {k[1]}: corr {np.corrcoef(ref, M[k])[0, 1]:.2f}, same sign {np.mean(np.sign(ref) == np.sign(M[k])):.2f}")
    print(f"{name} (against m2 0.9, gap 0.5): " + "; ".join(out), flush=True)

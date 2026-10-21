"""Does the tie margin M0 (score[p2] - score[p3] at m2 = m3) depend mainly on the pair (p2, p3), on relative ages, or on all three?
Deterministic, m2 = m3 = 0.9, m1 = 1.4, others at -0.46. Reports the share of M0's variance explained by group means."""
import itertools
import sys

import numpy as np

sys.path.insert(0, "/data/output/research/algzoo/20261016_run3/scripts")
from band_origin import T, b, scores  # noqa: E402

from algzoo.core import example, zoo  # noqa: E402

n = len(T); ar = np.arange(n)


def share(y, keys):
    keys = [tuple(k) for k in keys]
    u = {k: i for i, k in enumerate(sorted(set(keys)))}
    g = np.array([u[k] for k in keys])
    means = np.bincount(g, weights=y) / np.bincount(g)
    return 1 - np.var(y - means[g]) / np.var(y), len(u)


for name in sys.argv[1:] or ["example", 2, 3, 0, 4, 1]:
    p = example() if name == "example" else zoo(16, 10, int(name))
    X = np.full((n, 10), b); X[ar, T[:, 0]] = 1.4; X[ar, T[:, 1]] = 0.9; X[ar, T[:, 2]] = 0.9
    S = scores(p, X)
    M0 = S[ar, T[:, 1]] - S[ar, T[:, 2]]
    lo, hi = np.minimum(T[:, 1], T[:, 2]), np.maximum(T[:, 1], T[:, 2])
    sign = np.where(T[:, 1] < T[:, 2], 1, -1)      # orient: earlier minus later
    y = M0 * sign                                   # earlier-slot minus later-slot score at the tie
    out = [f"{name}: earlier-minus-later tie margin mean {y.mean():+.2f}, spread {y.std():.2f}"]
    for lab, keys in [("pair", np.c_[lo, hi]), ("pair ages (gap, steps after)", np.c_[hi - lo, 9 - hi]),
                      ("pair + largest before/between/after", np.c_[lo, hi, (T[:, 0] > lo).astype(int) + (T[:, 0] > hi)]),
                      ("all three", np.c_[T[:, 0], lo, hi])]:
        sh, k = share(y, keys)
        out.append(f"{lab} ({k} groups) {sh:.2f}")
    print("; ".join(out), flush=True)

# What sharing costs: band mass from per-placement tie margins versus from group means (pair + largest's relative position).
print("cost of sharing across placements (band = max(0, -M0) / slope, both ties' orientation kept):")
for name in sys.argv[1:] or ["example", 2, 3, 0, 4, 1]:
    p = example() if name == "example" else zoo(16, 10, int(name))
    X = np.full((n, 10), b); X[ar, T[:, 0]] = 1.4; X[ar, T[:, 1]] = 0.9; X[ar, T[:, 2]] = 0.9
    S = scores(p, X); M0 = S[ar, T[:, 1]] - S[ar, T[:, 2]]
    X[ar, T[:, 2]] = 0.895; S5 = scores(p, X); slope = np.median((S5[ar, T[:, 1]] - S5[ar, T[:, 2]] - M0) / 0.005)
    lo, hi = np.minimum(T[:, 1], T[:, 2]), np.maximum(T[:, 1], T[:, 2])
    for lab, keys in [("pair", np.c_[T[:, 1], T[:, 2]]), ("pair + largest before/between/after", np.c_[T[:, 1], T[:, 2], (T[:, 0] > lo).astype(int) + (T[:, 0] > hi)])]:
        kk = [tuple(k) for k in keys]; u = {k: i for i, k in enumerate(sorted(set(kk)))}; g = np.array([u[k] for k in kk])
        Mg = (np.bincount(g, weights=M0) / np.bincount(g))[g]
        full = np.maximum(0, -M0) / slope; shared = np.maximum(0, -Mg) / slope
        # convert mean band width to points of accuracy with the second-third gap density at zero (2.54 per unit)
        print(f"  {name} {lab}: mean band {full.mean():.4f} per placement, shared {shared.mean():.4f}; "
              f"difference {100 * 2.54 * (shared.mean() - full.mean()):+.2f} points of accuracy")

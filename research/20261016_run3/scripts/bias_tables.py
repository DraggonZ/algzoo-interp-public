"""Do the 720 near-tie biases come from a small table? (goal 1; deterministic)

Tie margin M(p1, p2, p3) = score[p2] - score[p3] at m2 = m3 = 0.9, m1 = 1.4, others -0.46. Models, fitted by least squares
over the 720 placements (in-sample variance explained, then leave-one-largest-position-out to check it is not memorizing):
  A: M = V(p1, p2) - V(p1, p3)                    (each vote's size depends on its slot and the largest's slot; 90 numbers)
  B: M = V_late(p1, p_late) - V_early(p1, p_early), signed by which of p2, p3 is later (180 numbers)
Also the variance explained by a single 10x10 table measured directly: V(p1, q) = score at q minus mean, for the largest at
p1 and one input of 0.9 at q (90 two-input runs), i.e. no fitting."""
import itertools
import sys

import numpy as np

sys.path.insert(0, "/data/output/research/algzoo/20261016_run3/scripts")
from band_origin import scores  # noqa: E402

from algzoo.core import example, zoo  # noqa: E402

T = np.array(list(itertools.permutations(range(10), 3))); n = len(T); ar = np.arange(n)
b = -0.46




for name in sys.argv[1:] or ["example", 2, 3, 0, 4, 1]:
    p = example() if name == "example" else zoo(16, 10, int(name))
    X = np.full((n, 10), b); X[ar, T[:, 0]] = 1.4; X[ar, T[:, 1]] = 0.9; X[ar, T[:, 2]] = 0.9
    S = scores(p, X); M = S[ar, T[:, 1]] - S[ar, T[:, 2]]
    out = [f"{name}: tie-margin spread {M.std():.2f}"]
    # model A
    D = np.zeros((n, 100))
    D[ar, T[:, 0] * 10 + T[:, 1]] += 1; D[ar, T[:, 0] * 10 + T[:, 2]] -= 1
    # model B
    late = np.maximum(T[:, 1], T[:, 2]); early = np.minimum(T[:, 1], T[:, 2]); sgn = np.where(T[:, 1] > T[:, 2], 1.0, -1.0)
    E = np.zeros((n, 200))
    E[ar, T[:, 0] * 10 + late] += sgn; E[ar, 100 + T[:, 0] * 10 + early] -= sgn
    for lab, Dm in [("A (vote size by own slot and largest's slot)", D), ("B (separate tables for later and earlier vote)", E)]:
        coef, *_ = np.linalg.lstsq(Dm, M, rcond=None)
        r2 = 1 - np.var(M - Dm @ coef) / np.var(M)
        # held out: leave out each largest position in turn is impossible (tables indexed by it); instead leave out each p3
        pred = np.zeros(n)
        for q in range(10):
            tr = T[:, 2] != q
            c, *_ = np.linalg.lstsq(Dm[tr], M[tr], rcond=None)
            pred[~tr] = Dm[~tr] @ c
        r2h = 1 - np.var(M - pred) / np.var(M)
        out.append(f"{lab}: explained {r2:.2f} in sample, {r2h:.2f} held out by third's slot")
    # direct table from two-input runs
    X2 = np.full((100, 10), b)
    for a in range(10):
        for q in range(10):
            X2[a * 10 + q, a] = 1.4
            if q != a:
                X2[a * 10 + q, q] = 0.9
    V = scores(p, X2).reshape(10, 10, 10)            # [largest, other, slot]
    Vd = np.array([[V[a, q, q] if q != a else 0 for q in range(10)] for a in range(10)])
    pd = Vd[T[:, 0], T[:, 1]] - Vd[T[:, 0], T[:, 2]]
    out.append(f"direct two-input table: correlation with the tie margin {np.corrcoef(pd, M)[0, 1]:.2f}")
    print("; ".join(out), flush=True)

"""How much of ARC's second-third mistakes does the per-placement mechanism explain list by list?
For each random list with three positive top numbers: scale by the second (homogeneity), so second = 1, largest = g,
third = s < 1. Predict 'answers the third' if, at the nearest grid value of g, the walked margin for that placement
(fillers fixed at -0.46) favours the third at s. Compare with ARC's answer, and split mismatches by the largest
filler relative to the second (f4 = fourth / second): kept-range (>= 0.75) or not."""
import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from pl_flip import walk  # noqa: E402

from algzoo.core import example  # noqa: E402

P = example(); wi, wh, wo = P
pl = np.array(list(itertools.permutations(range(10), 3))); N = len(pl); r = np.arange(N)
GRID = np.array([1.05, 1.15, 1.3, 1.6, 2.0, 3.0, 5.0])
# wrong intervals per (grid g, placement): list of (s_lo, s_hi) with third winning, s in [0.3, 1]
wrong = {}
for gi, g in enumerate(GRID):
    base = np.full((N, 10), -0.46); base[r, pl[:, 0]] = g; base[r, pl[:, 1]] = 1.0
    pieces, _ = walk(P, base, pl[:, 2], pl[:, 1], pl[:, 2], 1.0, 0.3)
    for i, pc in enumerate(pieces):
        iv = []
        for s0, s1, m0, m1 in pc:
            lo_s, hi_s = min(s0, s1), max(s0, s1)
            if m0 > 0 and m1 > 0:
                iv.append((lo_s, hi_s))
            elif (m0 > 0) != (m1 > 0):
                root = s0 + (s1 - s0) * m0 / (m0 - m1)
                iv.append((root, hi_s) if (m1 > 0) == (s1 > s0) else (lo_s, root))
        wrong[gi, tuple(pl[i])] = iv
rng = np.random.default_rng(3); X = rng.standard_normal((300_000, 10))
o = np.argsort(-X, 1); v = np.take_along_axis(X, o, 1)
keep = v[:, 2] > 0; X, o, v = X[keep], o[keep], v[keep]
h = np.zeros((len(X), 16))
for t in range(10):
    h = np.maximum(0, h @ wh.T + X[:, t:t + 1] * wi)
ans = (h @ wo.T).argmax(1)
actual = ans == o[:, 2]
g = v[:, 0] / v[:, 1]; s = v[:, 2] / v[:, 1]; f4 = v[:, 3] / v[:, 1]
gi = np.abs(np.log(g)[:, None] - np.log(GRID)[None]).argmin(1)
pred = np.array([any(a <= si <= b for a, b in wrong[k, (o[i, 0], o[i, 1], o[i, 2])]) for i, (k, si) in enumerate(zip(gi, s))])
print(f"{len(X)} lists with three positive numbers; ARC answers the third in {100*actual.mean():.2f}%, mechanism predicts {100*pred.mean():.2f}%")
print(f"of ARC's third-answers, predicted: {100*np.mean(pred[actual]):.1f}%; of predicted, ARC agrees: {100*np.mean(actual[pred]):.1f}%")
for lab, m in (("fourth >= 0.75 of second (could be kept)", f4 >= 0.75), ("fourth 0 to 0.75", (f4 > 0) & (f4 < 0.75)), ("fourth negative", f4 <= 0)):
    print(f"  {lab}: share {100*m.mean():.0f}%; ARC third-answers {100*actual[m].mean():.2f}%, predicted {100*pred[m].mean():.2f}%; "
          f"recall {100*np.mean(pred[m & actual]):.0f}%, precision {100*np.mean(actual[m & pred]) if (m & pred).any() else float('nan'):.0f}%")

# Which simplification loses the list-level agreement? Evaluate ARC directly on modified lists (one run each).
def third_answer(Xm):
    hh = np.zeros((len(Xm), 16))
    for t in range(10):
        hh = np.maximum(0, hh @ wh.T + Xm[:, t:t + 1] * wi)
    return (hh @ wo.T).argmax(1) == o[:, 2]


Xs = X / v[:, 1:2]                                           # second = 1
i = np.arange(len(X))
fill = np.ones_like(Xs, bool); fill[i, o[:, 0]] = fill[i, o[:, 1]] = fill[i, o[:, 2]] = False
Xa = Xs.copy(); Xa[fill] = -0.46                             # (a) exact g, fillers fixed
Xb = Xs.copy(); Xb[i, o[:, 0]] = GRID[gi]                    # (b) real fillers, g on the grid
Xc = Xa.copy(); Xc[i, o[:, 0]] = GRID[gi]                    # (c) both simplifications (the mechanism's prediction)
for lab, Xm in (("(a) exact largest, fillers fixed at -0.46", Xa), ("(b) real fillers, largest on the grid", Xb), ("(c) both simplifications", Xc)):
    p = third_answer(Xm)
    print(f"{lab}: rate {100*p.mean():.2f}%; recall {100*np.mean(p[actual]):.0f}%, precision {100*np.mean(actual[p]):.0f}%")

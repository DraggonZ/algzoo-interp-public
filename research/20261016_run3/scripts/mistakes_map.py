"""Where the full-input mistakes sit, from the three-input reduction's quadrature grids (reduce3_quad.py output).

The reduction matches the true accuracy within 0.07 points for ARC's model and seeds 0, 2, 3, 4, so its mistake map is a
map of the network's mistakes. Kinds: second at most 0; largest-second gap below 0.1; second-third gap below 0.1; other.
Also: which slot the network picks when wrong, and the arrival order of the three inputs.
"""
import itertools
import sys

import numpy as np
from scipy.special import ndtr, ndtri

sys.path.insert(0, "/data/output/research/algzoo/20261016_run3/scripts")
from reduce3_quad import M2, U, V, WU, WV  # noqa: E402

T = np.array(list(itertools.permutations(range(10), 3)))
P2 = ndtr(M2)
m1 = ndtri(P2[:, None] + U[None, :] * (1 - P2[:, None]))     # [N2, NU]
m3 = ndtri(V[None, :] ** (1 / 8) * P2[:, None])               # [N2, NV]
g12 = (m1 - M2[:, None])[:, :, None] * np.ones((1, 1, len(V)))
g23 = (M2[:, None] - m3)[:, None, :] * np.ones((1, len(U), 1))
m2g = M2[:, None, None] * np.ones_like(g12)
for name in sys.argv[1:] or ["example", "2", "3", "0", "4", "1"]:
    d = np.load(f"/data/output/research/algzoo/20261016_run3/reports/reduce3_quad_{name}.npz")
    r, wt, pred = d["right"], d["wt"], d["pred"]
    W = wt[None, :, None, None] * WU[None, None, :, None] * WV[None, None, None, :] * 100   # points
    wrong = ~r
    tot = (W * wrong).sum()
    neg = m2g <= 0
    n12 = (g12 < 0.1) & ~neg
    n23 = (g23 < 0.1) & ~neg
    kinds = {"second <= 0": neg, "largest-second < 0.1 only": n12 & ~n23, "second-third < 0.1 only": n23 & ~n12,
             "both gaps < 0.1": n12 & n23, "neither": ~neg & ~n12 & ~n23}
    print(f"== {name}: mistakes {tot:.2f} points")
    for k, m in kinds.items():
        mw = (W * (wrong & m[None])).sum()
        # answer when wrong, within this kind
        ans = {}
        for lab, sel in [("largest's slot", pred == T[:, 0, None, None, None]), ("third's slot", pred == T[:, 2, None, None, None]),
                         ("slot 9", (pred == 9) & (T[:, 0, None, None, None] != 9) & (T[:, 2, None, None, None] != 9))]:
            ans[lab] = (W * (wrong & m[None] & sel)).sum() / max(mw, 1e-12)
        print(f"  {k:28s} {mw:5.2f} points ({mw / tot:4.0%}); answers: " + ", ".join(f"{a} {v:.0%}" for a, v in ans.items()))
    # arrival order: role arriving last, and for second-third near-ties whether the later of the two wins
    last = np.argmax(T, 1)          # 0 largest, 1 second, 2 third arrives last
    for j, lab in enumerate(["largest", "second", "third"]):
        sel = last == j
        print(f"  {lab} arrives last: mistakes {(W * wrong)[sel].sum():.2f} points")
    m23 = wrong & n23[None] & ~n12[None]
    third_later = T[:, 2] > T[:, 1]
    a = (W * m23)[third_later].sum(); b = (W * m23)[~third_later].sum()
    print(f"  second-third near-tie mistakes: third arrives after the second {a:.2f} points, before {b:.2f}")
    m12 = wrong & n12[None] & ~n23[None]
    sec_later = T[:, 1] > T[:, 0]
    a = (W * m12)[sec_later].sum(); b = (W * m12)[~sec_later].sum()
    print(f"  largest-second near-tie mistakes: second arrives after the largest {a:.2f} points, before {b:.2f}")

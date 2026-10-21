"""Where does the two-spike case fail? Exact wrong intervals of the ratio, by order and gap, with probability mass."""
import numpy as np

from algzoo.core import example
from algzoo.exact_z2 import LO, HI, angle_mass, correct_intervals, pair_intervals

params = example()
tot_wrong = 0.0
rows = []
for p1 in range(10):
    for p2 in range(10):
        if p1 == p2:
            continue
        cells = pair_intervals(params, p1, p2, [0])
        good = correct_intervals(params, p1, p2, cells, [0])
        # wrong intervals = complement of good within (LO, HI)
        edges = [LO] + [e for g in good for e in g] + [HI]
        wrong = [(edges[i], edges[i + 1]) for i in range(0, len(edges), 2) if edges[i + 1] > edges[i] + 1e-12]
        for lo, hi in wrong:
            m = angle_mass(lo, hi)
            tot_wrong += m
            rows.append((p1, p2, np.tan(lo), np.tan(hi), m))
print(f"total wrong two-spike probability {tot_wrong:.4f} (1 - Z_2)")
rows.sort(key=lambda x: -x[4])
print("largest wrong intervals: largest pos, second pos, ratio range (second/largest), probability")
for p1, p2, a, b, m in rows[:25]:
    print(f"  {p1} {p2} {'max first' if p1 < p2 else '2nd first'}  ratio {a:+.3f} to {b:+.3f}   {m:.5f}")
by = {}
for p1, p2, a, b, m in rows:
    key = ("negative 2nd" if b <= 0 else "near-tie (ratio > 0.8)" if a >= 0.8 else "other",)
    by[key] = by.get(key, 0) + m
print("by kind:", {k[0]: round(v, 4) for k, v in by.items()})

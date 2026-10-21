"""Does an 8-slot ring confuse ages that differ by 8? Two-spike accuracy by the second's position, and the wrong answers."""
from collections import Counter

import numpy as np

from algzoo.core import example
from algzoo.exact_z2 import LO, HI, angle_mass, correct_intervals, pair_intervals

params = example()
w_hi, w_hh, w_oh = params
acc_by_p2 = np.zeros(10); mass_by_p2 = np.zeros(10)
wrong_ans = Counter()
for p1 in range(10):
    for p2 in range(10):
        if p1 == p2:
            continue
        cells = pair_intervals(params, p1, p2, [0])
        good = correct_intervals(params, p1, p2, cells, [0])
        acc_by_p2[p2] += sum(angle_mass(a, b) for a, b in good)
        mass_by_p2[p2] += angle_mass(LO, HI)
        edges = [LO] + [e for g in good for e in g] + [HI]
        for i in range(0, len(edges), 2):
            lo, hi = edges[i], edges[i + 1]
            if hi <= lo + 1e-12:
                continue
            th = 0.5 * (lo + hi)
            x = np.zeros(10); x[p1], x[p2] = np.cos(th), np.sin(th)
            h = np.zeros(16)
            for t in range(10):
                h = np.maximum(0, w_hh @ h + x[t] * w_hi)
            wrong_ans[(p2, int((w_oh @ h).argmax()))] += angle_mass(lo, hi)
print("two-spike accuracy by the second's position:", np.round(acc_by_p2 / mass_by_p2, 3))
print("largest wrong answers (second's position -> answer given: probability):")
for (p2, a), m in sorted(wrong_ans.items(), key=lambda kv: -kv[1])[:15]:
    print(f"  {p2} -> {a}: {m:.5f}")

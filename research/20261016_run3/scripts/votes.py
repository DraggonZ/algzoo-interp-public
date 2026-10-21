"""Two-spike case: split the final scores into the largest number's vote u and the second's vote v (exact, per cell).

Within a linear cell, scores = m1 * u + m2 * v with u = W_oh A[:,0], v = W_oh A[:,1]. Evaluated at ratio m2/m1 = r.
Spread = max minus min over positions, after removing the mean (only differences between positions matter).
"""
import sys

import numpy as np

from algzoo.core import example, zoo
from algzoo.exact_z2 import pair_intervals

name = sys.argv[1] if len(sys.argv) > 1 else "example"
r = float(sys.argv[2]) if len(sys.argv) > 2 else 0.7
params = example() if name == "example" else zoo(16, 10, int(name))
w_oh = params[2]
th = np.arctan2(r, 1.0)
rows = []
for p1 in range(10):
    for p2 in range(10):
        if p1 == p2:
            continue
        for lo, hi, A in pair_intervals(params, p1, p2, [0]):
            if lo <= th < hi:
                u, v = w_oh @ A[:, 0], w_oh @ A[:, 1]
                rows.append((p1, p2, u, v))
                break
print(f"model {name}, ratio {r}")
print("order  age(max) age(2nd) | spread u  spread r*v | argmax u, argmax v, answer (2nd at) ")
ok = 0
for p1, p2, u, v in rows:
    s = u + r * v
    order = "max first" if p1 < p2 else "2nd first"
    ok += s.argmax() == p2
    print(f"{order:9s} {9-p1:3d} {9-p2:3d}   | {np.ptp(u):7.2f} {np.ptp(r*v):8.2f}   | {u.argmax()} {v.argmax()} {s.argmax()} ({p2}) {'' if s.argmax()==p2 else 'WRONG'}")
print("right:", ok, "of", len(rows))
U = np.array([np.ptp(u) for *_, u, v in rows]); V = np.array([np.ptp(r * v) for *_, u, v in rows])
for lab, k in [("max first", np.array([p1 < p2 for p1, p2, *_ in rows])), ("2nd first", np.array([p1 > p2 for p1, p2, *_ in rows]))]:
    print(f"{lab}: median spread of max's vote {np.median(U[k]):.2f}, of 2nd's vote {np.median(V[k]):.2f}; argmax v = 2nd position in",
          sum(rows[i][3].argmax() == rows[i][1] for i in np.where(k)[0]), "of", k.sum())

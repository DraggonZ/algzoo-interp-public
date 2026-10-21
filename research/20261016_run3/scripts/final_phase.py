"""Two-spike final states: where is the wheel at the end, and whose age sets it?

Simulate m1 = 1 (largest) and m2 = r at every position pair. Project the final state on the wheel plane found from the
single-input response. Compare the final angle with the angle a single input would have at the largest's age and at the
second's age. Also test superposition: is the final state close to m1*H[age1] + m2*H[age2]?
"""
import sys

import numpy as np

from algzoo.core import example

r = float(sys.argv[1]) if len(sys.argv) > 1 else 0.7
w_hi, w_hh, w_oh = example()
h = np.maximum(0, w_hi); H = [h]
for k in range(1, 41):
    h = np.maximum(0, w_hh @ h); H.append(h)
H = np.array(H)
mu = H[1:].mean(0)
_, _, Vt = np.linalg.svd(H[1:] - mu, full_matrices=False)
P2 = Vt[:2]


def run(x):
    h = np.zeros(16)
    for t in range(10):
        h = np.maximum(0, w_hh @ h + x[t] * w_hi)
    return h


def angle(v, scale):
    p = P2 @ (v - scale * mu)
    return np.degrees(np.arctan2(p[1], p[0])), np.hypot(*p)


single = [angle(H[k], 1.0)[0] for k in range(10)]
print(f"ratio {r}. single-input wheel angle by age:", np.round(single, 0))
print("order      age1 age2 | final angle | single angle at age1, at age2 | superposition error (relative) | answer")
res = []
for p1 in range(10):
    for p2 in range(10):
        if p1 == p2:
            continue
        x = np.zeros(10); x[p1] = 1.0; x[p2] = r
        hf = run(x)
        a1, a2 = 9 - p1, 9 - p2
        ang, rad = angle(hf, 1.0 + r)
        sup = H[a1] + r * H[a2]
        err = np.linalg.norm(hf - sup) / np.linalg.norm(hf)
        ans = int(np.argmax(w_oh @ hf))
        res.append((p1 < p2, a1, a2, ang, single[a1], single[a2], err, ans == p2))
        if (p1 + p2) % 3 == 0:
            print(f"{'max first' if p1 < p2 else '2nd first':9s} {a1:4d} {a2:4d} | {ang:8.0f}   | {single[a1]:8.0f} {single[a2]:8.0f}   | {err:6.2f} | {'ok' if ans == p2 else 'WRONG'}")


def circ(a, b):
    return np.abs((np.asarray(a) - np.asarray(b) + 180) % 360 - 180)


for lab, sel in [("max first", True), ("2nd first", False)]:
    R = [x for x in res if x[0] == sel]
    d1 = circ([x[3] for x in R], [x[4] for x in R]); d2 = circ([x[3] for x in R], [x[5] for x in R])
    print(f"{lab}: median angle gap to single-input angle at age1 {np.median(d1):.0f} deg, at age2 {np.median(d2):.0f} deg; median superposition error {np.median([x[6] for x in R]):.2f}")

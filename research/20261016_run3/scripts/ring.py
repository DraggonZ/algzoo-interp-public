"""Is the network after the second input one fixed linear map that acts like a circular shift?

M* = step matrix with neurons 4, 5, 9 off (the most common pattern on quiet steps after two inputs).
1. Propagate the actual state right after the later input with M* only; compare the answer with the network's.
2. How close is M* to a circular shift? Powers: for k = 1..12, how well does (M*)^k map onto identity (cosine of
   vectorized matrices), and the readout of slot vectors.
"""
import numpy as np

from algzoo.core import example

w_hi, w_hh, w_oh = example()
on = np.ones(16); on[[4, 5, 9]] = 0
M = on[:, None] * w_hh
for r in (0.3, 0.7, 0.95):
    agree = right = n = 0
    for p1 in range(10):
        for p2 in range(10):
            if p1 == p2:
                continue
            x = np.zeros(10); x[p1] = 1.0; x[p2] = r
            h = np.zeros(16)
            t0 = max(p1, p2)
            for t in range(t0 + 1):
                h = np.maximum(0, w_hh @ h + x[t] * w_hi)
            hl = h.copy(); hn = h.copy()
            for t in range(t0 + 1, 10):
                hl = M @ hl
                hn = np.maximum(0, w_hh @ hn)
            a_l, a_n = (w_oh @ hl).argmax(), (w_oh @ hn).argmax()
            agree += a_l == a_n; right += a_l == p2; n += 1
    print(f"ratio {r}: fixed linear map after the later input: same answer as network {agree}/{n}, right {right}/{n}")
print("how close are powers of M* to the identity (cosine between matrices):")
I = np.eye(16)
for k in range(1, 13):
    Mk = np.linalg.matrix_power(M, k)
    print(f"  k={k:2d}: {np.sum(Mk * I) / np.linalg.norm(Mk) / 4:.2f}")

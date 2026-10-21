"""Review 2 check: does the free map h -> ReLU(W_hh h) keep rotating, or settle on a fixed direction?
Starts: post-write states of two-spike inputs (pair (0,1), (2,0) and (4,5), m2/m1 = 0.6) and random positive states."""
import numpy as np

from algzoo.core import example

w_hi, w_hh, w_oh = example()
rng = np.random.default_rng(1)
starts = []
for p1, p2 in ((0, 1), (2, 0), (4, 5)):
    h = np.zeros(16)
    for t in range(max(p1, p2) + 1):
        x = 1.0 if t == p1 else (0.6 if t == p2 else 0.0)
        h = np.maximum(w_hh @ h + x * w_hi, 0)
    starts.append(h)
starts += [np.abs(rng.standard_normal(16)) for _ in range(2)]
for h0 in starts:
    h = h0 / np.linalg.norm(h0)
    out = []
    for t in range(60):
        g = np.maximum(w_hh @ h, 0)
        nrm = np.linalg.norm(g)
        g /= nrm
        out.append((t, np.linalg.norm(g - h), nrm, int(np.argmax(w_oh @ g))))
        h = g
    print("step: |change of direction|, growth, readout argmax ->", [(t, round(c, 3), round(n, 3), a) for t, c, n, a in out if t in (0, 1, 2, 4, 8, 12, 20, 30, 59)])

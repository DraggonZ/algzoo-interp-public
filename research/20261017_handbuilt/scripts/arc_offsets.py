"""Does the trained network's store readout cancel an offset held by the cancellers (then a new largest at step 9,
which silences the cancellers, would leave a phantom) or one held by the memory (no phantom)? On lists without a new
largest at steps 8-9, fit each group's push on every position as slope * largest + intercept, averaged over lists."""
import numpy as np
from algzoo.core import example
wi, wh, wo = (np.asarray(w, dtype=float) for w in example()); wi = wi.reshape(-1)
X = np.random.default_rng(33).standard_normal((200_000, 10)); B, n = X.shape
h = np.zeros((B, 16))
for t in range(n): h = np.maximum(h @ wh.T + X[:, t:t + 1] * wi, 0)
m1p = np.stack([np.max(np.c_[np.zeros((B, 1)), X[:, :t]], 1) for t in range(n + 1)], 1)
rec = X > m1p[:, :n]; m1 = m1p[:, n]
groups = {"cancellers 1,6,7,8": [1, 6, 7, 8], "memory 0,3,10-15": [0, 3, 10, 11, 12, 13, 14, 15], "store 2": [2],
          "detector 4": [4], "keep 5": [5], "late 9": [9]}
np.set_printoptions(precision=1, suppress=True, linewidth=160)
for name, m in (("no new largest at 8 or 9", ~rec[:, 8] & ~rec[:, 9]), ("new largest at 9", rec[:, 9])):
    print(f"--- {name}: push per unit of the largest (slope over lists), positions 0-9")
    tot = 0
    for g, idx in groups.items():
        push = h[m][:, idx] @ wo[:, idx].T
        A = np.c_[m1[m], np.ones(m.sum())]
        slope = np.linalg.lstsq(A, push, rcond=None)[0][0]; tot = tot + slope
        print(f"  {g:20s} {slope}")
    print(f"  {'sum':20s} {tot}")

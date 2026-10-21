"""Back-to-back new largests at t, t+1 (t = 3..5, none later). Does the trained network's cancellers' final push keep a
wave of the first one (x_t), which no longer is the largest? Fit each group's push on each position as
a * x_t + b * x_{t+1} + c, and compare with lists whose only late new largest is a single one at t+1 (old largest P)."""
import numpy as np
from algzoo.core import example
wi, wh, wo = (np.asarray(w, dtype=float) for w in example()); wi = wi.reshape(-1)
X = np.random.default_rng(34).standard_normal((600_000, 10)); B, n = X.shape
m1p = np.stack([np.max(np.c_[np.zeros((B, 1)), X[:, :t]], 1) for t in range(n)], 1); rec = X > m1p
h = np.zeros((B, 16)); H = []
for t in range(n): h = np.maximum(h @ wh.T + X[:, t:t + 1] * wi, 0); H.append(h)
groups = {"cancellers": [1, 6, 7, 8], "memory": [0, 3, 10, 11, 12, 13, 14, 15], "store": [2], "keep 5": [5], "late 9": [9]}
np.set_printoptions(precision=1, suppress=True, linewidth=170)
t = 4
later = rec[:, t + 2:].any(1)
for name, m, xa in (("back to back at 4, 5", rec[:, t] & rec[:, t + 1] & ~later, X[:, t]),
                    ("single at 5, old largest P earlier", ~rec[:, t] & rec[:, t + 1] & ~later, m1p[:, t + 1])):
    print(f"--- {name} ({m.sum()} lists): coefficient on the earlier value (x_4 or P) / on x_5, positions 0-9")
    for g, idx in groups.items():
        push = h[m][:, idx] @ wo[:, idx].T
        A = np.c_[xa[m], X[m, t + 1], np.ones(m.sum())]
        co = np.linalg.lstsq(A, push, rcond=None)[0]
        print(f"  {g:10s} earlier {co[0]}\n  {'':10s} x_5     {co[1]}")

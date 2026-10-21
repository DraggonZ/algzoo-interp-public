"""Follow-up to hiding_steps.py: each hiding neuron's pre-activation k steps after a record, with negative numbers in
between, is about c (ref - x) with ref = (a p + b m1) / (a + b), a weighted average of the previous largest p and the
new largest m1. Report, for k = 1..5, the share of the new largest in each neuron's reference, b / (a + b), and the
sum a + b against c (the weight on the arriving number)."""
import numpy as np
from algzoo.core import example
wi, wh, wo = (np.asarray(p, float) for p in example())
rng = np.random.default_rng(16); X = rng.standard_normal((4000000, 10)); N = len(X)
h = np.zeros((N, 16)); PRE = []
for t in range(10): pre = h @ wh.T + X[:, t:t + 1] * wi; h = np.maximum(pre, 0); PRE.append(pre)
print("share of the new largest in each neuron's reference (sum of the two weights / weight on x)")
for k in range(1, 6):
    rows = []
    for r in range(2, 10 - k):
        p = np.maximum(X[:, :r].max(1), 0); ok = (X[:, r] > p + 0.05) & (p > 0.2)
        if k > 1: ok &= (X[:, r + 1:r + k] < 0).all(1)
        rows.append(np.c_[p[ok], X[ok, r], X[ok, r + k], PRE[r + k][ok][:, [1, 6, 7, 8]]])
    R = np.concatenate(rows); A = R[:, :3]; out = []
    for i, n in enumerate((1, 6, 7, 8)):
        a, b, c = np.linalg.lstsq(A, R[:, 3 + i], rcond=None)[0]
        out.append(f"{n}: {b / (a + b):5.2f} ({(a + b) / -c:4.2f})")
    print(f"  {k} step(s) after (n={len(R):7d}): " + "  ".join(out))

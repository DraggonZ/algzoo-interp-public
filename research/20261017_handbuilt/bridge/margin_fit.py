"""Follow-up to margin_source.py: on lists where a record at r is overtaken at q by a number less than 5% larger (which
stays largest), fit the trained network's margin M = score[r] - score[q] as a x_r + b (x_q - x_r) + c, per placement.
If c is small and a is large, the trained network pushes the new largest's position below the revealed record by an
amount proportional to their value, not to the jump. Also the share of these lists the trained network gets right."""
import numpy as np
from algzoo.core import example
rng = np.random.default_rng(17); X = rng.standard_normal((3000000, 10)); N = len(X)
wi, wh, wo = (np.asarray(p, float) for p in example())
def scores(X):
    h = np.zeros((len(X), 16))
    for t in range(10): h = np.maximum(h @ wh.T + X[:, t:t + 1] * wi, 0)
    return h @ wo.T
for r, q in [(1, 4), (2, 5), (3, 6), (4, 7), (2, 8), (5, 8)]:
    pre = np.maximum(X[:, :r].max(1), 0)
    ok = (X[:, r] > pre) & (X[:, r] > 0.3) & (X[:, r + 1:q].max(1) < X[:, r]) & (X[:, q] > X[:, r]) & (X[:, q] < 1.05 * X[:, r])
    if q < 9: ok &= X[:, q + 1:].max(1) < X[:, r]
    L = scores(X[ok]); M = L[:, r] - L[:, q]; xr = X[ok, r]; dq = X[ok, q] - xr
    A = np.c_[xr, dq, np.ones(ok.sum())]; c = np.linalg.lstsq(A, M, rcond=None)[0]; e = M - A @ c
    right = (L.argmax(1) == r).mean()
    print(f"r{r} q{q} (n={ok.sum()}): M = {c[0]:5.2f} x_r {c[1]:+6.2f} (x_q - x_r) {c[2]:+5.2f}  (variance matched {1 - e.var() / M.var():.2f}); right {right:.2f}")

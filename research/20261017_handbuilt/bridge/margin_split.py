"""Which score moves? Same lists as margin_fit.py (record at r overtaken at q by under 5%). Per unit, median over lists:
(a) the jump grows (x_q nudged): change of score[q] and of score[r];
(b) both values grow together (x_r and x_q nudged, jump fixed): change of score[r] and of score[q].
Compare with a kept number's own score per unit (vote_profile.log: 23-42)."""
import numpy as np
from algzoo.core import example
rng = np.random.default_rng(17); X = rng.standard_normal((3000000, 10))
wi, wh, wo = (np.asarray(p, float) for p in example())
def scores(X):
    h = np.zeros((len(X), 16))
    for t in range(10): h = np.maximum(h @ wh.T + X[:, t:t + 1] * wi, 0)
    return h @ wo.T
eps = 1e-3
for r, q in [(1, 4), (2, 5), (3, 6), (4, 7)]:
    pre = np.maximum(X[:, :r].max(1), 0)
    ok = (X[:, r] > pre) & (X[:, r] > 0.3) & (X[:, r + 1:q].max(1) < X[:, r]) & (X[:, q] > X[:, r]) & (X[:, q] < 1.05 * X[:, r])
    if q < 9: ok &= X[:, q + 1:].max(1) < X[:, r]
    Xs = X[ok]; L0 = scores(Xs)
    Xa = Xs.copy(); Xa[:, q] += eps; La = (scores(Xa) - L0) / eps
    Xb = Xs.copy(); Xb[:, q] += eps; Xb[:, r] += eps; Lb = (scores(Xb) - L0) / eps
    print(f"r{r} q{q}: jump grows: score[q] {np.median(La[:, q]):7.1f}, score[r] {np.median(La[:, r]):7.1f} | both grow: score[r] {np.median(Lb[:, r]):7.1f}, score[q] {np.median(Lb[:, q]):7.1f}")

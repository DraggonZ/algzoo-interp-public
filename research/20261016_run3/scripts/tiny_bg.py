"""Tiny model with a background: the other eight inputs at b instead of 0.

Predicted scores = background-only scores + m1 * (lone largest on the background minus background-only, per unit) at its
age + c * m2 at the second's slot, with c from one network step on top of a lone input's state on the background.
Inputs: top two of 10 Gaussians kept, the other eight set to b. Agreement with the network on those inputs.
"""
import sys

import numpy as np

from algzoo.core import example, zoo

b = float(sys.argv[1]) if len(sys.argv) > 1 else -0.5


def run(params, X):
    w_hi, w_hh, w_oh = params
    h = np.zeros((X.shape[0], len(w_hi)))
    for t in range(10):
        h = np.maximum(0, h @ w_hh.T + X[:, t, None] * w_hi)
    s = h @ w_oh.T
    return s - s.mean(1, keepdims=True)


rng = np.random.default_rng(5)
X = rng.standard_normal((100000, 10))
o = np.argsort(X, 1); p1, p2 = o[:, -1], o[:, -2]
idx = np.arange(len(X)); m1, m2 = X[idx, p1], X[idx, p2]
keep = (m2 > b)
X2 = np.full_like(X, b); X2[idx, p1] = m1; X2[idx, p2] = m2
for name in ["example", 0, 1, 2, 3, 4]:
    params = tuple(np.asarray(q, np.float64) for q in (example() if name == "example" else zoo(16, 10, name)))
    S0 = run(params, np.full((1, 10), b))[0]
    # lone largest of size 1 at each position, on the background, minus background (scale-free approximation)
    Lp = np.zeros((10, 10))
    for p in range(10):
        x = np.full((1, 10), b); x[0, p] = 1.0
        Lp[p] = run(params, x)[0] - S0
    # vote per unit for a second arriving after the largest (largest at 0, second at 5, value 0.5 vs 0)
    xa = np.full((2, 10), b); xa[:, 0] = 1.0; xa[0, 5] = 0.5; xa[1, 5] = 0.0
    sa = run(params, xa)
    v = (sa[0] - sa[1]) / 0.5; c = v[5] - np.delete(v, 5).mean()
    net = run(params, X2).argmax(1)
    S = S0[None] + m1[:, None] * Lp[p1]
    S[idx, p2] += c * (m2 - 0.0)
    pred = S.argmax(1)
    right = net == p2
    k = keep
    print(f"{name}: background {b}: network right {right[k].mean():.4f}; tiny model agrees {np.mean(pred[k] == net[k]):.4f}; "
          f"predicts mistakes {np.mean(pred[k & ~right] != p2[k & ~right]):.2f}; false alarms {np.mean(pred[k & right] != p2[k & right]):.4f}; c {c:.0f}")

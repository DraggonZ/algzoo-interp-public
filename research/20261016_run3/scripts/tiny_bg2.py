"""Tiny model on a constant background b (the typical value of the eight smaller inputs), built properly.

Predicted scores = (scores with the largest alone on the background, at its actual value and position)
                   + c * m2 at the second's slot,
with c = per-unit vote of a second arriving after a largest of size 1 on the same background (averaged over gaps).
This tests whether the two inputs' effects add on a background. Inputs: top two of 10 Gaussians, the rest set to b.
"""
import sys

import numpy as np

from algzoo.core import example, zoo

b = float(sys.argv[1]) if len(sys.argv) > 1 else -0.32


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
X2 = np.full_like(X, b); X2[idx, p1] = m1; X2[idx, p2] = m2
X1 = np.full_like(X, b); X1[idx, p1] = m1
keep = m2 > b
for name in ["example", 0, 1, 2, 3, 4]:
    params = tuple(np.asarray(q, np.float64) for q in (example() if name == "example" else zoo(16, 10, name)))
    cs = []
    for g in range(1, 10):
        for q1 in range(0, 10 - g):
            xa = np.full((2, 10), b); xa[:, q1] = 1.0; xa[0, q1 + g] = 0.6; xa[1, q1 + g] = b
            sa = run(params, xa); v = (sa[0] - sa[1]) / (0.6 - b)
            cs.append(v[q1 + g] - np.delete(v, q1 + g).mean())
    c = float(np.median(cs))
    net = run(params, X2).argmax(1)
    S = run(params, X1)
    S[idx, p2] += c * (m2 - b)
    pred = S.argmax(1)
    right = net == p2; k = keep
    print(f"{name}: network right {right[k].mean():.4f}; tiny model agrees {np.mean(pred[k] == net[k]):.4f}; "
          f"mistakes predicted {np.mean(pred[k & ~right] != p2[k & ~right]):.2f}; false alarms {np.mean(pred[k & right] != p2[k & right]):.4f}; c {c:.0f}")

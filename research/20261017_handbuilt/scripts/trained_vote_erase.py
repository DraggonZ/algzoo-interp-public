"""Does a later number erase an earlier small number's vote in ARC's model?

Base: largest 2.0 at p1, second 1.0 at p2 (zero background). Small a at i, then
b at j > i. Vote of a = score at slot i with a minus without a (minus the mean
change over slots). Compare b = 0, b < a, b > a (b below the second).
"""
import itertools

import numpy as np
import torch as th

from algzoo.core import example, forward

P = example()
a = 0.5
res = {}
for bval in (0.0, 0.3, 0.7, 0.9):
    vs = []
    for p1, p2, i, j in itertools.permutations(range(10), 4):
        if j <= i:
            continue
        x = np.zeros(10)
        x[p1], x[p2], x[j] = 2.0, 1.0, bval
        xa = x.copy()
        xa[i] = a
        s = forward(P, th.tensor(np.stack([x, xa]))).numpy()
        d = s[1] - s[0]
        d -= d.mean()
        vs.append((d[i] / a, i, j, p1, p2))
    v = np.array(vs)
    res[bval] = v
    print(f"b={bval}: vote of a per unit, median {np.median(v[:,0]):.1f}, mean {v[:,0].mean():.1f}, 10th pct {np.percentile(v[:,0],10):.1f}")
# split by whether the max arrives between a and b etc.
for bval in (0.3, 0.7):
    v = res[bval]
    for name, mask in [("max after b", v[:, 3] > v[:, 2]), ("max between", (v[:, 3] > v[:, 1]) & (v[:, 3] < v[:, 2])), ("max before a", v[:, 3] < v[:, 1])]:
        print(f"  b={bval} {name:13s}: median vote {np.median(v[mask,0]):.1f}  (b=0 same cases: {np.median(res[0.0][mask,0]):.1f})")

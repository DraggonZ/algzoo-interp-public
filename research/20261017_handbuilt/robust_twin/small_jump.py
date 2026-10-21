"""Lists whose second-largest came before the largest (an overtaken record), binned by the largest's relative lead over
the second, (m1 - m2) / m1: accuracy of the trained network, v18 and v19 (kappa 0.5) at trained scale, and v19 with an
idealized perfect wipe. Shows whether failed wipes cost the twins on lists the trained network gets right."""
import numpy as np
from algzoo.handbuilt_v19 import build_v19
import simlib

X = simlib.lists(79, 200_000); y = X.argsort(1)[:, -2]; o = X.argsort(1)[:, ::-1]; T = [3, 4, 5, 6]
cm = np.maximum.accumulate(X, 1); prev = np.c_[np.zeros((len(X), 1)), cm[:, :-1]]; rec = X > np.maximum(prev, 0)
m1 = np.take_along_axis(X, o[:, :1], 1)[:, 0]; m2 = np.take_along_axis(X, o[:, 1:2], 1)[:, 0]
before = o[:, 1] < o[:, 0]; rel = (m1 - m2) / np.maximum(m1, 1e-9)
def ans(P, perfect=False):
    h = np.zeros((len(X), 16))
    for t in range(10):
        h = np.maximum(h @ P[1].T + X[:, t:t + 1] * P[0], 0)
        if perfect: h[np.ix_(rec[:, t], T)] = 0.0
    return (h @ P[2].T).argmax(1)
kw = dict(g=10, a=10, w=10, lam=10.0, c=-9.0, eps=0.1)
nets = {"trained": simlib.run(X)[0].argmax(1),
        "v18 trained scale": ans(tuple(np.asarray(p, float) for p in build_v19(kappa=0.0, **kw).params)),
        "v19 kappa 0.5": ans(tuple(np.asarray(p, float) for p in build_v19(kappa=0.5, **kw).params)),
        "v19 kappa 0.5, perfect wipe": ans(tuple(np.asarray(p, float) for p in build_v19(kappa=0.5, **kw).params), True)}
bins = [0, 0.02, 0.05, 0.1, 0.2, 0.4, 1.01]
print("bins of (m1 - m2)/m1, second before the largest:", " ".join(f"{a}-{b}" for a, b in zip(bins[:-1], bins[1:])))
for name, a in nets.items():
    row = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = before & (rel >= lo) & (rel < hi); row.append(f"{np.mean(a[m] == y[m]):.3f}")
    print(f"  {name:28s}: " + " ".join(row) + f"   (all such lists {np.mean(a[before] == y[before]):.3f}; second after the largest {np.mean(a[~before] == y[~before]):.3f})")
print("share of such lists per bin:", " ".join(f"{np.mean(before & (rel >= lo) & (rel < hi)):.3f}" for lo, hi in zip(bins[:-1], bins[1:])))

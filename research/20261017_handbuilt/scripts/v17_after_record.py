"""Accuracy when the second sits right after the largest (the largest never overtaken), by position; and the
trained network's margin for those lists, regressed on y (second), p (record before the largest), x_L, z (third)."""
import numpy as np, torch as th, sys
sys.path.insert(0, "scripts")
from algzoo.core import example, forward
from handbuilt_v16 import build_v16, scores
X = np.random.default_rng(32).standard_normal((600_000, 10)); B = len(X)
m1p = np.stack([np.max(np.c_[np.zeros((B, 2)), X[:, :t]], 1) for t in range(10)], 1); rec = X > m1p
o = X.argsort(1); sec, lar, thi = o[:, -2], o[:, -1], o[:, -3]
T = forward(example(), th.tensor(X)).numpy(); V = scores(build_v16(gamma=0.25), X)
m = (sec == lar + 1)
print("second right after the largest: share", m.mean().round(3), "trained", (T.argmax(1) == sec)[m].mean().round(3), "v16", (V.argmax(1) == sec)[m].mean().round(3))
m2 = (sec == lar + 2); print("second two after the largest: share", m2.mean().round(3), "trained", (T.argmax(1) == sec)[m2].mean().round(3), "v16", (V.argmax(1) == sec)[m2].mean().round(3))
for name, S in (("trained", T), ("v16 x100", 100 * V)):
    for gap, mm in ((1, m), (3, sec == lar + 3)):
        i = np.where(mm & (sec <= 7) & (thi < lar))[0]
        own = S[i, sec[i]]; F = np.c_[X[i, sec[i]], X[i, lar[i]], m1p[i, lar[i]], X[i, thi[i]], np.ones(len(i))]
        c, *_ = np.linalg.lstsq(F, own - S[i, thi[i]], rcond=None)
        print(f"  {name:9s} second {gap} after the largest, third before it: margin = " + " ".join(f"{k} {v:+.1f}" for k, v in zip(("y", "x_L", "p", "third", "1"), c)))

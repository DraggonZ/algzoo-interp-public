"""Lists whose second is an overtaken record P at r (r <= 7), split by where the third sits relative to r."""
import numpy as np, torch as th, sys
sys.path.insert(0, "scripts")
from algzoo.core import example, forward
from handbuilt_v16 import build_v16, scores
X = np.random.default_rng(30).standard_normal((300_000, 10)); B = len(X)
m1p = np.stack([np.max(np.c_[np.zeros((B, 2)), X[:, :t]], 1) for t in range(10)], 1); rec = X > m1p
o = X.argsort(1); sec, lar, thi = o[:, -2], o[:, -1], o[:, -3]
base = rec[np.arange(B), sec] & (sec <= 7)
res = {"trained": forward(example(), th.tensor(X)).numpy().argmax(1), "v16": scores(build_v16(gamma=0.25), X).argmax(1)}
for off in (-2, -1, 1, 2, 3):
    m = base & (thi - sec == off) & (lar != thi)
    print(f"third at P{off:+d} (share {m.mean():.3f}): " + "  ".join(f"{k} {(v[m] == sec[m]).mean():.3f}" for k, v in res.items()))
m = base & (lar - sec == 1); print(f"current largest right after P (share {m.mean():.3f}): " + "  ".join(f"{k} {(v[m] == sec[m]).mean():.3f}" for k, v in res.items()))

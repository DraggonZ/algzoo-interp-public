"""Middle ground: memory gets the value wave (late gate), the clock a mix (value share beta, rest the jump), dip w
smaller so waves are only 1/(1-w) times a vote. Split by kind of list."""
import numpy as np, itertools
from handbuilt_v9 import build_v7, scores
X = np.random.default_rng(30).standard_normal((100_000, 10)); B = len(X)
m1p = np.stack([np.max(np.c_[np.zeros((B, 2)), X[:, :t]], 1) for t in range(10)], 1); rec = X > m1p
o = X.argsort(1); sec, lar, thi = o[:, -2], o[:, -1], o[:, -3]
cons = (rec[:, :-1] & rec[:, 1:]).any(1); onepos = (X > 0).sum(1) <= 1
ordinary = ~rec[:, 8] & ~rec[:, 9] & ~cons & ~onepos
res = []
for beta, w in itertools.product((1.0, 0.7, 0.5, 0.3), (0.8, 0.6, 0.4)):
    P = scores(build_v7(clock="jump", beta=beta, mem_jump=False, w=w), X).argmax(1); ok = P == sec; bad = ~ok
    res.append((ok.mean(), beta, w, ok[ordinary].mean(), ok[rec[:, 9]].mean(), ok[rec[:, 8]].mean(), ok[cons].mean(), (P == thi)[bad].mean(), (P == lar)[bad].mean()))
for r in sorted(res, reverse=True):
    print("all {:.3f}  beta {} w {}  ord {:.3f} rec9 {:.3f} rec8 {:.3f} b2b {:.3f} | wrong third {:.2f} largest {:.2f}".format(*r))

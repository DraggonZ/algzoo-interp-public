"""v15 step 1: v14 (jump-driven clock, jump waves in the memory, keep gate writes (1 - c) x previous largest at each new
largest) with the jump wave F times the spike, so an overtaken largest outscores the current one by (F - 1) x its own jump.
Two timings: jump into memory one step after the new largest (no dip, steady level), or two steps after via the late gate
(after a dip of depth w)."""
import numpy as np, itertools
from handbuilt_v9 import build_v7, scores
X = np.random.default_rng(30).standard_normal((200_000, 10)); B = len(X)
m1p = np.stack([np.max(np.c_[np.zeros((B, 2)), X[:, :t]], 1) for t in range(10)], 1); rec = X > m1p
o = X.argsort(1); sec, lar, thi = o[:, -2], o[:, -1], o[:, -3]
cons = (rec[:, :-1] & rec[:, 1:]).any(1); onepos = (X > 0).sum(1) <= 1
kinds = {"onepos": onepos, "last": rec[:, 9] & ~onepos, "step8": rec[:, 8] & ~rec[:, 9] & ~onepos,
         "row": cons & ~rec[:, 8] & ~rec[:, 9] & ~onepos, "ord": ~onepos & ~rec[:, 9] & ~rec[:, 8] & ~cons}
def show(name, **kw):
    P = scores(build_v7(**kw), X).argmax(1); ok = P == sec; bad = ~ok
    return (ok.mean(), name, *[ok[m].mean() for m in kinds.values()], (P == thi)[bad].mean(), (P == lar)[bad].mean())
res = []
for F in (1.0, 1.3, 1.6, 2.0, 3.0):
    res.append(show(f"one step, F {F}", clock="jump", beta=0.0, mem_jump=True, w=0.0, lam_s=10.0, keep_rec=True, F=F))
    for w in (0.3, 0.6):
        res.append(show(f"two steps, dip {w}, F {F}", clock="jump", beta=0.0, mem_jump=False, late_jump=True, keep_rec=True, w=w, F=F * (1 - w)))
res.sort(reverse=True)
print("trained: all 0.953 | onepos 0.224 last 0.944 step8 0.973 row 0.962 ord 0.962 | wrong third 0.61 largest 0.15")
for r in res[:8]: print("all {:.3f}  {:26s} | onepos {:.3f} last {:.3f} step8 {:.3f} row {:.3f} ord {:.3f} | wrong third {:.2f} largest {:.2f}".format(*r))

import numpy as np
from handbuilt_v9 import build_v7, scores
X = np.random.default_rng(30).standard_normal((200_000, 10)); B = len(X)
m1p = np.stack([np.max(np.c_[np.zeros((B, 2)), X[:, :t]], 1) for t in range(10)], 1); rec = X > m1p
o = X.argsort(1); sec, lar, thi = o[:, -2], o[:, -1], o[:, -3]
cons = (rec[:, :-1] & rec[:, 1:]).any(1); onepos = (X > 0).sum(1) <= 1
ordinary = ~rec[:, 8] & ~rec[:, 9] & ~cons & ~onepos; secrec = rec[np.arange(B), sec]
def show(name, **kw):
    P = scores(build_v7(**kw), X).argmax(1); ok = P == sec; bad = ~ok
    print(f"{name:40s} all {ok.mean():.3f} ord {ok[ordinary].mean():.3f} (second revealed {ok[ordinary & secrec].mean():.3f}, kept {ok[ordinary & ~secrec].mean():.3f}) "
          f"rec9 {ok[rec[:, 9]].mean():.3f} b2b {ok[cons].mean():.3f} | wrong: third {(P == thi)[bad].mean():.2f} largest {(P == lar)[bad].mean():.2f}")
show("v7 late clock, dip w 0.8", clock="late")
for w in (0.5, 0.3, 0.0):
    show(f"v7 late clock, dip w {w}", clock="late", w=w)
show("v7 late clock, no clock-level dip at all", clock="late", w=0.0, clock_dip=False, lam_s=10.0)

import numpy as np, sys
from handbuilt_v9 import build_v7, scores
X = np.random.default_rng(30).standard_normal((200_000, 10)); B = len(X)
m1p = np.stack([np.max(np.c_[np.zeros((B, 2)), X[:, :t]], 1) for t in range(10)], 1); rec = X > m1p
sec = X.argsort(1)[:, -2]; cons = (rec[:, :-1] & rec[:, 1:]).any(1); onepos = (X > 0).sum(1) <= 1
ordinary = ~rec[:, 8] & ~rec[:, 9] & ~cons & ~onepos
def show(name, **kw):
    ok = scores(build_v7(**kw), X).argmax(1) == sec
    print(f"{name:34s} all {ok.mean():.3f}  ordinary {ok[ordinary].mean():.3f}  rec9 {ok[rec[:, 9]].mean():.3f}  rec8 {ok[rec[:, 8]].mean():.3f}  b2b {ok[cons].mean():.3f}")
show("v7 late clock (reference)", clock="late")
for beta in (1.0, 0.5, 0.3, 0.0):
    show(f"jump clock, value share {beta}, no dip", clock="jump", beta=beta, w=0.0, lam_s=10.0)

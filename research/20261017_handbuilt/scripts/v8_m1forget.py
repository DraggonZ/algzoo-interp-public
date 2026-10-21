import numpy as np
from handbuilt_v8 import build_v7, scores
X = np.random.default_rng(30).standard_normal((200_000, 10)); B = len(X)
m1p = np.stack([np.max(np.c_[np.zeros((B, 2)), X[:, :t]], 1) for t in range(10)], 1); rec = X > m1p
o = X.argsort(1); sec, lar, thi = o[:, -2], o[:, -1], o[:, -3]
cons = (rec[:, :-1] & rec[:, 1:]).any(1); onepos = (X > 0).sum(1) <= 1
ordinary = ~rec[:, 8] & ~rec[:, 9] & ~cons & ~onepos
def show(name, **kw):
    net = build_v7(**kw); P = scores(net, X).argmax(1); ok = P == sec; bad = ~ok
    print(f"{name:36s} ({len(net.params[0])}) all {ok.mean():.3f} ord {ok[ordinary].mean():.3f} rec9 {ok[rec[:, 9]].mean():.3f} rec8 {ok[rec[:, 8]].mean():.3f} "
          f"b2b {ok[cons].mean():.3f} | wrong: third {(P == thi)[bad].mean():.2f} largest {(P == lar)[bad].mean():.2f} 8/9 {np.isin(P, [8, 9])[bad].mean():.2f}")
show("v7 + z + second store (17)", clock="z")
for c1 in (0.0, 0.1, 0.25, 0.5, 1.0):
    show(f"z, forget vs largest c1={c1}", clock="z", second=False, c1=c1)

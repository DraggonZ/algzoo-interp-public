import numpy as np
from handbuilt_v7 import build_v7, scores
X = np.random.default_rng(30).standard_normal((200_000, 10)); B = len(X)
m1p = np.stack([np.max(np.c_[np.zeros((B, 2)), X[:, :t]], 1) for t in range(10)], 1); rec = X > m1p
sec = X.argsort(1)[:, -2]; cons = (rec[:, :-1] & rec[:, 1:]).any(1)
for G in (1e4, 1e3, 300, 100, 30, 10):
    net = build_v7(a=min(1000, G), g=G, g2=G, K=min(1000, G), lam_c=min(1000, G))
    ok = scores(net, X).argmax(1) == sec; W = np.abs(np.concatenate([p.ravel() for p in net.params])).max()
    print(f"gain {G:7.0f} (largest weight {W:8.0f}): all {ok.mean():.3f}  rec 9 {ok[rec[:, 9]].mean():.3f}  rec 8 {ok[rec[:, 8]].mean():.3f}  b2b {ok[cons].mean():.3f}")

import numpy as np
from handbuilt_v9 import build_v7, scores
X = np.random.default_rng(30).standard_normal((200_000, 10)); B = len(X)
m1p = np.stack([np.max(np.c_[np.zeros((B, 2)), X[:, :t]], 1) for t in range(10)], 1); rec = X > m1p
sec = X.argsort(1)[:, -2]; one = (rec.sum(1) == 1) & ((X > 0).sum(1) >= 2) & ~rec[:, 8:].any(1)
np.set_printoptions(precision=2, suppress=True, linewidth=150)
for beta in (1.0, 0.0):
    net = build_v7(clock="jump", beta=beta, w=0.0, lam_s=10.0); ok = scores(net, X).argmax(1) == sec
    print(f"value share {beta}: lists with a single new largest (second is a kept number): {ok[one].mean():.3f}")
    x = np.full(10, -0.5); x[2] = 1.0; x[5] = 0.6
    print("   largest 1.0 at 2, kept 0.6 at 5:", scores(net, x[None])[0])

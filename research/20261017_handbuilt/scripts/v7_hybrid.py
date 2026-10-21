import numpy as np
from handbuilt_v7 import build_v7, scores
X = np.random.default_rng(30).standard_normal((200_000, 10)); B, n = X.shape
z = np.zeros((B, 2)); m1p = np.stack([np.max(np.c_[z, X[:, :t]], 1) for t in range(n)], 1)
rec = X > m1p; sec = X.argsort(1)[:, -2]; onepos = (X > 0).sum(1) <= 1; cons = (rec[:, :-1] & rec[:, 1:]).any(1)
for al in (0, 0.25, 0.5, 0.75, 1):
    ok = scores(build_v7(alpha=al), X).argmax(1) == sec
    print(f"share rung up {al:.2f}: all {ok.mean():.3f}  rec 9 {ok[rec[:, 9]].mean():.3f}  rec 8 {ok[rec[:, 8]].mean():.3f}  b2b {ok[cons].mean():.3f}")

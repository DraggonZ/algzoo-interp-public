import numpy as np
from handbuilt_v6 import build_v6, scores
X = np.random.default_rng(28).standard_normal((300_000, 10)); B, n = X.shape
P = scores(build_v6(), X).argmax(1)
o = X.argsort(1); sec, lar = o[:, -2], o[:, -1]; r = np.arange(B)
z = np.zeros((B, 2)); m1p = np.stack([np.max(np.c_[z, X[:, :t]], 1) for t in range(n)], 1)
rec = X > m1p
ok = P == sec
print("by position of the second:", [round(ok[sec == p].mean(), 2) for p in range(10)])
print("second is revealed old largest:", ok[rec[r, sec]].mean().round(3), " kept:", ok[~rec[r, sec]].mean().round(3))
print("answer position histogram when wrong:", np.bincount(P[~ok], minlength=10))
print("answer is the largest when wrong:", (P[~ok] == lar[~ok]).mean().round(2))

import numpy as np
from handbuilt_v7 import build_v7, scores
X = np.random.default_rng(28).standard_normal((300_000, 10)); B, n = X.shape
S = scores(build_v7(), X); P = S.argmax(1)
o = X.argsort(1); sec, lar, thi = o[:, -2], o[:, -1], o[:, -3]; r = np.arange(B)
z = np.zeros((B, 2)); m1p = np.stack([np.max(np.c_[z, X[:, :t]], 1) for t in range(n)], 1)
m2p = np.stack([np.sort(np.c_[z, X[:, :t]], 1)[:, -2] for t in range(n)], 1)
rec = X > m1p; ok = P == sec
print("second revealed old largest:", ok[rec[r, sec]].mean().round(3), "| second kept:", ok[~rec[r, sec]].mean().round(3))
print("wrong answers: largest", (P[~ok] == lar[~ok]).mean().round(2), "third", (P[~ok] == thi[~ok]).mean().round(2))
# ideal relative votes
c = 0.75
v = np.where(rec, 0.0, np.maximum((1 - c) * X - c * np.maximum(m2p - X, 0), 0))
print("second's ideal vote vs its actual score (kept seconds, first 8 lists):")
idx = np.flatnonzero(~rec[r, sec] & ~ok)[:8]
np.set_printoptions(precision=2, suppress=True, linewidth=150)
for i in idx:
    print("  x", np.round(X[i], 2), "| sec", sec[i], "answer", P[i]); print("    ideal", np.round(v[i], 2)); print("    score", np.round(S[i], 2))

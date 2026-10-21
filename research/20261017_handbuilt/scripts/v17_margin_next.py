"""Margin score(P) - score(y) when the third y sits right after the revealed old largest P (at r); regress on
P, p (record before P), y, x_L (current largest). Trained network and v16."""
import numpy as np, torch as th, sys
sys.path.insert(0, "scripts")
from algzoo.core import example, forward
from handbuilt_v16 import build_v16, scores
X = np.random.default_rng(31).standard_normal((600_000, 10)); B = len(X)
m1p = np.stack([np.max(np.c_[np.zeros((B, 2)), X[:, :t]], 1) for t in range(10)], 1); rec = X > m1p
o = X.argsort(1); sec, lar, thi = o[:, -2], o[:, -1], o[:, -3]
m = rec[np.arange(B), sec] & (sec <= 6) & (thi == sec + 1) & (lar > sec + 1)
X = X[m]; s = sec[m]; l = lar[m]; t = thi[m]; i = np.arange(len(X))
P = X[i, s]; p = m1p[m][i, s]; y = X[i, t]; xl = X[i, l]
for name, S in (("trained", forward(example(), th.tensor(X)).numpy()), ("v16 x100", 100 * scores(build_v16(gamma=0.25), X))):
    mg = S[i, s] - S[i, t]
    F = np.c_[P, p, y, xl, np.ones(len(X))]
    c, *_ = np.linalg.lstsq(F, mg, rcond=None); r2 = 1 - ((F @ c - mg) ** 2).mean() / mg.var()
    print(f"{name:9s} n {len(X)} right {(mg > 0).mean():.3f} | margin = " + " ".join(f"{k} {v:+.1f}" for k, v in zip(("P", "p", "y", "x_L", "1"), c)) + f"  R2 {r2:.2f}")

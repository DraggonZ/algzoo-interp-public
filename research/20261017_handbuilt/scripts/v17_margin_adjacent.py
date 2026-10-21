"""Margin score(P) - score(x_L) when the current largest x_L sits right after the revealed old largest P;
regressed on P, p (record before P), x_L. Trained network and v16 (x100)."""
import numpy as np, torch as th, sys
sys.path.insert(0, "scripts")
from algzoo.core import example, forward
from handbuilt_v16 import build_v16, scores
X = np.random.default_rng(34).standard_normal((600_000, 10)); B = len(X)
m1p = np.stack([np.max(np.c_[np.zeros((B, 2)), X[:, :t]], 1) for t in range(10)], 1); rec = X > m1p
o = X.argsort(1); sec, lar = o[:, -2], o[:, -1]
for gap in (1, 2, 4):
    m = rec[np.arange(B), sec] & (lar == sec + gap) & (lar <= 7)
    Z = X[m]; s = sec[m]; l = lar[m]; i = np.arange(len(Z))
    P = Z[i, s]; p = m1p[m][i, s]; xl = Z[i, l]
    for name, S in (("trained", forward(example(), th.tensor(Z)).numpy()), ("v16 x100", 100 * scores(build_v16(gamma=0.25), Z))):
        mg = S[i, s] - S[i, l]; F = np.c_[P, p, xl, np.ones(len(Z))]
        c, *_ = np.linalg.lstsq(F, mg, rcond=None)
        print(f"gap {gap} {name:9s} right {(mg > 0).mean():.3f} margin = " + " ".join(f"{k} {v:+.1f}" for k, v in zip(("P", "p", "x_L", "1"), c)))

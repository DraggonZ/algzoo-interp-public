"""Final-score response to raising the revealed old largest P (at r, overtaken at L) by eps, at each offset q - r,
for the trained network and v16. Random lists where P is the second; P + eps stays below x_L."""
import numpy as np, torch as th, sys
sys.path.insert(0, "scripts")
from algzoo.core import example, forward
from handbuilt_v16 import build_v16, scores
rng = np.random.default_rng(5); X = rng.standard_normal((400_000, 10)); B = len(X)
m1p = np.stack([np.max(np.c_[np.zeros((B, 2)), X[:, :t]], 1) for t in range(10)], 1); rec = X > m1p
o = X.argsort(1); sec, lar = o[:, -2], o[:, -1]
ok = rec[np.arange(B), sec] & (X[np.arange(B), lar] - X[np.arange(B), sec] > 0.05)
X = X[ok]; sec = sec[ok]; lar = lar[ok]; B = len(X); eps = 0.02
X2 = X.copy(); X2[np.arange(B), sec] += eps
nets = {"trained": lambda Z: forward(example(), th.tensor(Z)).numpy(), "v16": lambda Z: scores(build_v16(gamma=0.25), Z)}
for name, f in nets.items():
    dS = (f(X2) - f(X)) / eps
    print(name)
    for gap in ((1,), (2,), (3, 4, 5, 6, 7, 8, 9)):
        m = np.isin(lar - sec, gap)
        row = []
        for off in range(-3, 4):
            q = sec + off; v = (q >= 0) & (q <= 9) & m & (q != lar)
            row.append(f"{off:+d}: {np.median(dS[v, q[v]]):6.2f}")
        print(f"  current largest {gap[0] if len(gap)==1 else '3+'} after P: " + "  ".join(row), f"| at current largest {np.median(dS[m, lar[m]]):6.2f}")

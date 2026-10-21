import numpy as np, sys
from algzoo.handbuilt_v23 import build_v23, scores
from algzoo.handbuilt_v19 import build_v19, scores as sc19
rng = np.random.default_rng(2); X = rng.standard_normal((400000, 10)); o = np.argsort(-X, 1); L, S = o[:, 0], o[:, 1]
np.set_printoptions(precision=0, suppress=True, linewidth=180)
for name, sc in (("v23", scores(build_v23(), X)), ("v19", sc19(build_v19(kappa=0.5), X))):
    ok = sc.argmax(1) == S; M = np.full((10, 10), np.nan)
    for l in range(10):
        for s_ in range(10):
            m = (L == l) & (S == s_)
            if m.any(): M[l, s_] = 100 * ok[m].mean()
    print(name, f"{ok.mean():.4f}; rows largest position, columns second position, % right"); print(M)

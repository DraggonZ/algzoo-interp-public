import numpy as np, sys
from algzoo.handbuilt_v24 import build_v24, scores
rng = np.random.default_rng(2); X = rng.standard_normal((400000, 10)); o = np.argsort(-X, 1); L, S, T3 = o[:, 0], o[:, 1], o[:, 2]
np.set_printoptions(precision=0, suppress=True, linewidth=180)
kw = eval(sys.argv[1]) if len(sys.argv) > 1 else {}
y = scores(build_v24(**kw), X).argmax(1); ok = y == S; M = np.full((10, 10), np.nan); ML = np.full((10, 10), np.nan)
for l in range(10):
    for s_ in range(10):
        m = (L == l) & (S == s_)
        if m.any(): M[l, s_] = 100 * ok[m].mean(); ML[l, s_] = 100 * (y[m] == l).mean()
print(kw, f"accuracy {ok.mean():.4f}; rows largest position, columns second position: % right"); print(M); print("% naming the largest"); print(ML)

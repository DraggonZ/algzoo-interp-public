"""Question: what do trained memory neurons 3 and 15 hold that v18's 16 neurons do not? Fit each (steps 5 and 9) as a
linear combination of v18's 16 neurons (trained gain scale) on 100,000 lists; on 100,000 others, take the residual and
ask which candidate quantity explains most of it (squared correlation), from a menu of named quantities of the list
so far. Reference: the same for memory neuron 10 (well matched)."""
import numpy as np
from algzoo.core import example
from algzoo.handbuilt_v18 import build_v18
rng = np.random.default_rng(6); X = rng.standard_normal((200000, 10)); N = len(X)
def states(P, X):
    wi, wh, wo = (np.asarray(p, float) for p in P); h = np.zeros((len(X), len(wi))); hs = []
    for t in range(10): h = np.maximum(h @ wh.T + X[:, t:t + 1] * wi, 0); hs.append(h)
    return np.stack(hs, 1)
HT = states(example(), X); H18 = states(build_v18(g=10, a=10, w=10, lam=10.0, c=-9.0, eps=0.1).params, X)
tr, te = slice(0, 100000), slice(100000, None)
def menu(t):
    xs = X[:, :t + 1]; ar = np.arange(t + 1)
    o = np.argsort(xs, 1); i1, i2 = o[:, -1], o[:, -2]
    m1 = np.maximum(xs.max(1), 0); m2 = np.maximum(np.take_along_axis(xs, i2[:, None], 1)[:, 0], 0)
    q = {"x now": X[:, t], "relu x now": np.maximum(X[:, t], 0), "x one step ago": X[:, t - 1], "relu x one step ago": np.maximum(X[:, t - 1], 0),
         "largest": m1, "second-largest": m2, "third-largest": np.maximum(np.take_along_axis(xs, o[:, -3:-2], 1)[:, 0], 0),
         "steps since largest": t - i1, "steps since second": t - i2, "largest x steps since": m1 * (t - i1), "second x steps since": m2 * (t - i2),
         "number right after largest": np.where(i1 < t, np.maximum(X[np.arange(N), np.minimum(i1 + 1, 9)], 0), 0),
         "previous record": None, "count of positive numbers": (xs > 0).sum(1), "sum of positive numbers": np.maximum(xs, 0).sum(1)}
    pm = np.maximum.accumulate(np.c_[np.zeros(N), np.maximum(xs, 0)], 1)[:, :-1]; rec = xs > pm
    lastrec = np.where(rec, ar, -1).max(1); prevval = pm[np.arange(N), np.maximum(lastrec, 0)]
    q["previous record"] = np.where(lastrec >= 0, prevval, 0)
    for k in range(t + 1):
        q[f"relu x at {k}"] = np.maximum(X[:, k], 0)
    return q
for t in (5, 9):
    Q = menu(t)
    for n in (3, 15, 10):
        A = np.c_[H18[:, t], np.ones(N)]; c = np.linalg.lstsq(A[tr], HT[tr, t, n], rcond=None)[0]
        res = HT[te, t, n] - A[te] @ c; frac = res.var() / HT[te, t, n].var()
        sc = sorted(((np.corrcoef(v[te], res)[0, 1] ** 2, k) for k, v in Q.items() if v[te].std() > 0), reverse=True)[:4]
        print(f"step {t}, trained {n:2d}: residual {frac:.2f} of variance; best single quantities for the residual:",
              "; ".join(f"{k} {s:.2f}" for s, k in sc))

"""r14a look 3: which arriving numbers should the push average over? Candidates with a reading:
 below2: x_new drawn from N(0,1) truncated below the list's second-largest (the arrival does not change the answer)
 below1: truncated below the list's largest
 neg: only non-positive arrivals (x_new = min(x, 0)); pos: x_new = max(x, 0)
Compared with zero (r12a) and the 50/50 mixture of zero and fresh (look 2's best). Seeds 101, 102."""
import numpy as np
from scipy.stats import norm
import r12a_x as X
from r14a_look1 import Var, mk, WH, WI

N = 400_000
for seed in (101, 102):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((N, 11))
    u = rng.uniform(size=N)
    h = np.zeros((N, 16))
    for t in range(10):
        h = np.maximum(h @ WH.T + x[:, t, None] * WI[None, :], 0)
    G = h.T @ h / N
    pre = h @ WH.T
    srt = np.sort(x[:, :10], 1)

    def M(xn):
        return np.linalg.solve(G, h.T @ np.maximum(pre + xn[:, None] * WI[None, :], 0) / N)
    trunc = lambda b: norm.ppf(u * norm.cdf(b))
    M0 = M(np.zeros(N)); M1 = M(x[:, 10])
    cands = {"zero": M0, "mix .5": (M0 + M1) / 2, "below2": M(trunc(srt[:, -2])), "below1": M(trunc(srt[:, -1])),
             "neg": M(np.minimum(x[:, 10], 0)), "pos": M(np.maximum(x[:, 10], 0))}
    out = []
    for k, Mk in cands.items():
        net, d = X.price(Var(mk(Mk)))
        out.append(f"{k} {net:+.1f} (g {d['C'][0]:.3f})")
    print(seed, " | ".join(out), flush=True)

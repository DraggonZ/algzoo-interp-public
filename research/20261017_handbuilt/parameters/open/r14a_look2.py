"""r14a look 2: the arriving number at the push's extra step. Target E over x_new ~ N(0, s^2) (s = 0 is r12a's choice,
s = 1 the task's own distribution), and mixtures p * fresh + (1 - p) * zero. Three list seeds."""
import sys
import numpy as np
import torch as th
import r12a_x as X
from r14a_look1 import Var, mk, R, WH, WI

N = 400_000
for seed in (101, 102, 103):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((N, 11))
    h = np.zeros((N, 16))
    for t in range(10):
        h = np.maximum(h @ WH.T + x[:, t, None] * WI[None, :], 0)
    G = h.T @ h / N
    pre = h @ WH.T
    def M(hn):
        return np.linalg.solve(G, h.T @ hn / N)
    M0 = M(np.maximum(pre, 0))
    out = []
    for s in (0.0, 0.3, 0.5, 0.7, 1.0):
        Ms = M(np.maximum(pre + s * x[:, 10, None] * WI[None, :], 0))
        out.append((f"s={s}", X.price(Var(mk(Ms)))[0]))
    M1 = M(np.maximum(pre + x[:, 10, None] * WI[None, :], 0))
    for p in (0.25, 0.5, 0.75):
        out.append((f"p={p}", X.price(Var(mk(p * M1 + (1 - p) * M0)))[0]))
    # half the time a number arrives: equivalently fresh number with prob 1/2 per list
    print(seed, " ".join(f"{a} {b:+.1f}" for a, b in out), flush=True)

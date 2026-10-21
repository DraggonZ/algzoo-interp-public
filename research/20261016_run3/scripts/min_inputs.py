"""How many small inputs (besides the top two) are needed to reproduce the network's answer?

For random inputs: answer on the full input vs on reduced inputs that keep the top two plus a subset of the other
inputs at their values (the rest zero). Level 0: top two alone gives the same answer. Level 1: some single extra input
does. Level 2: some pair does. Otherwise 3 or more. Reported separately for inputs the network gets right and wrong.
"""
import itertools
import sys

import numpy as np
import torch as th

from algzoo.bench import load_bench
from algzoo.core import example, zoo

dev = "cuda"


def answers(params, X):
    w = [th.as_tensor(p, device=dev, dtype=th.float32) for p in params]
    h = th.zeros(X.shape[0], w[0].shape[0], device=dev)
    for t in range(10):
        h = th.relu(h @ w[1].T + X[:, t, None] * w[0])
    return (h @ w[2].T).argmax(-1)


def levels(params, N=20000, seed=3):
    g = th.Generator(device=dev).manual_seed(seed)
    X = th.randn(N, 10, device=dev, generator=g)
    o = X.argsort(-1)
    top2 = o[:, -2:]; low = o[:, :8]
    full = answers(params, X)
    right = full == o[:, -2]
    base = th.zeros_like(X).scatter(1, top2, X.gather(1, top2))
    lvl = th.full((N,), 3, device=dev)
    a0 = answers(params, base)
    lvl[a0 == full] = 0
    hit1 = th.zeros(N, dtype=th.bool, device=dev)
    for i in range(8):
        Xi = base.clone(); idx = low[:, i:i + 1]
        Xi.scatter_(1, idx, X.gather(1, idx))
        hit1 |= answers(params, Xi) == full
    lvl[(lvl == 3) & hit1] = 1
    hit2 = th.zeros(N, dtype=th.bool, device=dev)
    for i, j in itertools.combinations(range(8), 2):
        Xi = base.clone(); idx = low[:, [i, j]]
        Xi.scatter_(1, idx, X.gather(1, idx))
        hit2 |= answers(params, Xi) == full
    lvl[(lvl == 3) & hit2] = 2
    lvl[lvl == 3] = 4
    hit3 = th.zeros(N, dtype=th.bool, device=dev)
    for i, j, k in itertools.combinations(range(8), 3):
        Xi = base.clone(); idx = low[:, [i, j, k]]
        Xi.scatter_(1, idx, X.gather(1, idx))
        hit3 |= answers(params, Xi) == full
    lvl[(lvl == 4) & hit3] = 3
    out = {}
    for lab, m in [("right", right), ("wrong", ~right)]:
        L = lvl[m].cpu().numpy()
        out[lab] = (m.float().mean().item(), [np.mean(L == k) for k in range(5)])
    return out


names, acc, models = load_bench(("mult",))
cases = [("ARC example", example())] + [(f"seed {s}", zoo(16, 10, s)) for s in range(5)] + \
        [(str(n), m) for n, m in zip(names, models) if str(n) in ("mult0.01_3", "mult0.01_8", "mult0.02_8", "mult0.02_16")]
for lab, p in cases:
    o = levels(p)
    print(f"{lab:12s}: right {o['right'][0]:.3f} -> levels 0/1/2/3/4+ {np.round(o['right'][1], 3)}; wrong -> {np.round(o['wrong'][1], 3)}")
    sys.stdout.flush()

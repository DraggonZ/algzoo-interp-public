"""Robustness check of memory_as_slots2.py: other lists (seed 21), and v18 at both gain settings (10 and 10,000).
Variance of each trained memory neuron matched by v18's slots plus store, median over steps 3-9."""
import numpy as np
from algzoo.core import example
from algzoo.handbuilt_v18 import build_v18
rng = np.random.default_rng(21); X = rng.standard_normal((200000, 10))
def states(P, X):
    wi, wh, wo = (np.asarray(p, float) for p in P); h = np.zeros((len(X), len(wi))); hs = []
    for t in range(10): h = np.maximum(h @ wh.T + X[:, t:t + 1] * wi, 0); hs.append(h)
    return np.stack(hs, 1)
HT = states(example(), X)
def r2(A, b):
    A1 = np.c_[A, np.ones(len(A))]; tr, te = slice(0, 100000), slice(100000, None)
    c = np.linalg.lstsq(A1[tr], b[tr], rcond=None)[0]; e = b[te] - A1[te] @ c
    return 1 - e.var() / b[te].var()
for name, kw in (("gains 10", dict(g=10, a=10, w=10, lam=10.0, c=-9.0, eps=0.1)), ("gains 10,000", {})):
    net = build_v18(**kw); I = net.idx; H = states(net.params, X); sl = [I[f"mem{j}"] for j in range(8)] + [I["store"]]
    print(f"v18 {name}: " + ", ".join(f"{n} {np.median([r2(H[:, t, sl], HT[:, t, n]) for t in range(3, 10)]):.2f}" for n in [10, 11, 0, 12, 13, 14, 15, 3]))

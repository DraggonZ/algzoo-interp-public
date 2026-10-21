"""Question: is the trained memory a linear re-coding of the hand-built delay line? Run the trained network and v18
(trained gain scale) on the same lists. For each trained memory neuron, at steps 5 and 9, the share of its variance
matched by a linear combination of (a) v18's eight memory slots plus its store, (b) all 16 v18 neurons; least
squares on 100,000 lists, scored on 100,000 others. Analysis of the trained network only: no hand-built weight comes
from this. Reference: the same for the trained hiding neurons."""
import numpy as np
from algzoo.core import example
from algzoo.handbuilt_v18 import build_v18
rng = np.random.default_rng(5); X = rng.standard_normal((200000, 10))
def states(P, X):
    wi, wh, wo = (np.asarray(p, float) for p in P); h = np.zeros((len(X), len(wi))); hs = []
    for t in range(10): h = np.maximum(h @ wh.T + X[:, t:t + 1] * wi, 0); hs.append(h)
    return np.stack(hs, 1)
HT = states(example(), X); net = build_v18(g=10, a=10, w=10, lam=10.0, c=-9.0, eps=0.1); I = net.idx; HV = states(net.params, X)
slots = [I[f"mem{j}"] for j in range(8)] + [I["store"]]
def r2(A, b):
    A1 = np.c_[A, np.ones(len(A))]; tr, te = slice(0, 100000), slice(100000, None)
    c = np.linalg.lstsq(A1[tr], b[tr], rcond=None)[0]; e = b[te] - A1[te] @ c
    return 1 - e.var() / b[te].var()
for t in (5, 9):
    print(f"step {t}")
    for n in [0, 3, 10, 11, 12, 13, 14, 15, 1, 6, 7, 8]:
        print(f"   trained {n:2d}: from v18 slots + store {r2(HV[:, t, slots], HT[:, t, n]):.2f}; from all 16 v18 neurons {r2(HV[:, t], HT[:, t, n]):.2f}")

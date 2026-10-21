"""Controls for memory_as_slots.py. For every trained neuron, steps 3-9, variance matched by a linear combination of:
  v18 (16 neurons, trained gain scale); v18's slots plus store; v5 (another hand-built 16-neuron network, built on a
  different memory and gate design); raw inputs so far, their positive parts, and the running largest (a plain
  baseline of 21 features); and v18 with its lists shuffled (unrelated lists: should give about 0).
Fitted on 100,000 lists, scored on 100,000 others; median over steps 3-9."""
import numpy as np
from algzoo.core import example
from algzoo.handbuilt_v18 import build_v18
from algzoo.handbuilt import build_v5
rng = np.random.default_rng(5); X = rng.standard_normal((200000, 10))
def states(P, X):
    wi, wh, wo = (np.asarray(p, float) for p in P); h = np.zeros((len(X), len(wi))); hs = []
    for t in range(10): h = np.maximum(h @ wh.T + X[:, t:t + 1] * wi, 0); hs.append(h)
    return np.stack(hs, 1)
HT = states(example(), X)
n18 = build_v18(g=10, a=10, w=10, lam=10.0, c=-9.0, eps=0.1); I = n18.idx; H18 = states(n18.params, X)
v5 = build_v5(); H5 = states(v5.params if hasattr(v5, "params") else v5, X)
slots = [I[f"mem{j}"] for j in range(8)] + [I["store"]]
def base(t):
    Xt = np.where(np.arange(10) <= t, X, 0.0)
    return np.c_[Xt, np.maximum(Xt, 0), np.maximum(np.maximum.accumulate(X, 1)[:, t], 0)]
def r2(A, b):
    A1 = np.c_[A, np.ones(len(A))]; tr, te = slice(0, 100000), slice(100000, None)
    c = np.linalg.lstsq(A1[tr], b[tr], rcond=None)[0]; e = b[te] - A1[te] @ c
    return 1 - e.var() / max(b[te].var(), 1e-12)
perm = rng.permutation(len(X))
srcs = {"v18 all": lambda t: H18[:, t], "v18 slots+store": lambda t: H18[:, t, slots], "v5 all": lambda t: H5[:, t],
        "inputs (21)": base, "v18 shuffled": lambda t: H18[perm, t]}
print("trained neuron | " + " | ".join(f"{k:15s}" for k in srcs))
for n in [2, 4, 5, 9, 1, 6, 7, 8, 0, 3, 10, 11, 12, 13, 14, 15]:
    vals = [np.median([r2(f(t), HT[:, t, n]) for t in range(3, 10)]) for f in srcs.values()]
    print(f"   {n:2d}          | " + " | ".join(f"{v:15.2f}" for v in vals))

"""Follow-up: are trained neurons 3 and 15 a clipped linear re-coding of v18? For each trained memory neuron, at steps
3-9: share of steps it is off (zero); variance of its pre-activation (input before the ReLU) matched by v18's 16
neurons at the same step; and accuracy of predicting on/off from the sign of that fitted pre-activation.
Fitted on 100,000 lists, scored on 100,000 others; median over steps."""
import numpy as np
from algzoo.core import example
from algzoo.handbuilt_v18 import build_v18
rng = np.random.default_rng(6); X = rng.standard_normal((200000, 10)); N = len(X)
def states(P, X, pre=False):
    wi, wh, wo = (np.asarray(p, float) for p in P); h = np.zeros((len(X), len(wi))); hs = []; ps = []
    for t in range(10):
        p = h @ wh.T + X[:, t:t + 1] * wi; ps.append(p); h = np.maximum(p, 0); hs.append(h)
    return np.stack(hs, 1), np.stack(ps, 1)
HT, PT = states(example(), X); H18, _ = states(build_v18(g=10, a=10, w=10, lam=10.0, c=-9.0, eps=0.1).params, X)
tr, te = slice(0, 100000), slice(100000, None)
print("neuron | off share | pre-activation matched | on/off predicted | post-activation matched (linear)")
for n in [0, 3, 10, 11, 12, 13, 14, 15]:
    off, r2p, agree, r2h = [], [], [], []
    for t in range(3, 10):
        A = np.c_[H18[:, t], np.ones(N)]
        c = np.linalg.lstsq(A[tr], PT[tr, t, n], rcond=None)[0]; e = PT[te, t, n] - A[te] @ c
        r2p.append(1 - e.var() / PT[te, t, n].var()); agree.append(((A[te] @ c > 0) == (PT[te, t, n] > 0)).mean())
        off.append((HT[te, t, n] == 0).mean())
        c2 = np.linalg.lstsq(A[tr], HT[tr, t, n], rcond=None)[0]; e2 = HT[te, t, n] - A[te] @ c2; r2h.append(1 - e2.var() / HT[te, t, n].var())
    print(f"  {n:2d}   |   {np.median(off):.2f}    |        {np.median(r2p):.2f}            |      {np.median(agree):.2f}        |  {np.median(r2h):.2f}")

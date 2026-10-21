"""Question: is neuron 3 poorly matched because its input is a small difference of large, well-matched terms?
(a) Size of neuron 3's pre-activation against the summed sizes of its terms w[3, j] h_j (and the input term), steps
3-9 pooled; same for the other memory neurons. (b) Neuron 3's pre-activation rebuilt from the trained weights applied
to each input neuron's best linear match from v18 (fitted on other lists): variance matched."""
import numpy as np
from algzoo.core import example
from algzoo.handbuilt_v18 import build_v18
rng = np.random.default_rng(6); X = rng.standard_normal((200000, 10)); N = len(X)
wi, wh, wo = (np.asarray(p, float) for p in example())
def states(P, X):
    a, b, _ = (np.asarray(p, float) for p in P); h = np.zeros((len(X), len(a))); hs = [h]; ps = []
    for t in range(10):
        p = h @ b.T + X[:, t:t + 1] * a; ps.append(p); h = np.maximum(p, 0); hs.append(h)
    return np.stack(hs, 1), np.stack(ps, 1)            # hs[:, t] is the state before step t
HT, PT = states(example(), X); H18, _ = states(build_v18(g=10, a=10, w=10, lam=10.0, c=-9.0, eps=0.1).params, X)
tr, te = slice(0, 100000), slice(100000, None)
print("(a) pre-activation size / summed size of its terms (median over lists, steps 3-9)")
for n in [0, 3, 10, 11, 12, 13, 14, 15]:
    r = []
    for t in range(3, 10):
        terms = HT[:, t] * wh[n][None, :]; tot = terms.sum(1) + wi[n] * X[:, t]
        r.append(np.median(np.abs(tot) / (np.abs(terms).sum(1) + np.abs(wi[n] * X[:, t]) + 1e-9)))
    print(f"   neuron {n:2d}: {np.median(r):.2f}")
print("(b) neuron 3's pre-activation from the trained weights applied to v18-matched inputs")
for t in (5, 7, 9):
    A = np.c_[H18[:, t], np.ones(N)]             # v18 state before step t
    Hm = np.zeros((N, 16))
    for j in range(16):
        c = np.linalg.lstsq(A[tr], HT[tr, t, j], rcond=None)[0]; Hm[:, j] = A @ c
    pre_hat = Hm @ wh[3] + wi[3] * X[:, t]; e = PT[te, t, 3] - pre_hat[te]
    print(f"   step {t}: variance matched {1 - e.var() / PT[te, t, 3].var():.2f}")

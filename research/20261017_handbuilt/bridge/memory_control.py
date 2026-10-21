"""Control for memory_as_slots2.py (review's cheap check): is the match of the trained memory by v18's delay line
specific to a delay line, or would any 8-dimensional memory fed the same writes do? Memories, all fed v18's newest-slot
value (what enters slot 0 each step) and read together with v18's store:
  v18's delay line (slots shift by one each step);
  a random rotating memory: m_t = Q m_(t-1) + e1 * write_t, Q a random orthogonal 8x8 matrix (three seeds);
  a decaying sum: m_t = 0.8 m_(t-1) + write_t repeated in 8 copies with decay 0.5-0.95 (a leaky integrator bank).
Variance of each trained memory neuron matched (linear fit on 100,000 lists, scored on 100,000), median over steps 3-9."""
import numpy as np
from algzoo.core import example
from algzoo.handbuilt_v18 import build_v18
rng = np.random.default_rng(5); X = rng.standard_normal((200000, 10)); N = len(X)
def states(P, X):
    wi, wh, wo = (np.asarray(p, float) for p in P); h = np.zeros((len(X), len(wi))); hs = []
    for t in range(10): h = np.maximum(h @ wh.T + X[:, t:t + 1] * wi, 0); hs.append(h)
    return np.stack(hs, 1)
HT = states(example(), X); net = build_v18(g=10, a=10, w=10, lam=10.0, c=-9.0, eps=0.1); I = net.idx; H = states(net.params, X)
slots = [I[f"mem{j}"] for j in range(8)]; S = I["store"]; write = H[:, :, slots[0]]
def r2(A, b):
    A1 = np.c_[A, np.ones(len(A))]; tr, te = slice(0, 100000), slice(100000, None)
    c = np.linalg.lstsq(A1[tr], b[tr], rcond=None)[0]; e = b[te] - A1[te] @ c
    return 1 - e.var() / b[te].var()
def bank(step):
    m = np.zeros((N, 8)); ms = []
    for t in range(10): m = step(m, write[:, t]); ms.append(m)
    return np.stack(ms, 1)
mems = {"v18 delay line": H[:, :, slots]}
for sd in (1, 2, 3):
    Q = np.linalg.qr(np.random.default_rng(sd).standard_normal((8, 8)))[0]
    mems[f"random rotation {sd}"] = bank(lambda m, w, Q=Q: m @ Q.T + np.outer(w, np.eye(8)[0]))
dec = np.linspace(0.5, 0.95, 8); mems["leaky bank"] = bank(lambda m, w: m * dec[None, :] + w[:, None])
names = [0, 3, 10, 11, 12, 13, 14, 15]
print("trained memory neuron:      " + "  ".join(f"{n:5d}" for n in names))
for k, M in mems.items():
    vals = [np.median([r2(np.c_[M[:, t], H[:, t, S]], HT[:, t, n]) for t in range(3, 10)]) for n in names]
    print(f"{k:26s} " + "  ".join(f"{v:5.2f}" for v in vals))

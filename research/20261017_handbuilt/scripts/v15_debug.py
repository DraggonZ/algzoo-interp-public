import numpy as np
from handbuilt_v9 import build_v7, scores
np.set_printoptions(precision=2, suppress=True, linewidth=170)
net = build_v7(clock="jump", beta=0.0, mem_jump=False, late_jump=True, keep_rec=True, w=0.3, F=2 * 0.7)
X = np.random.default_rng(31).standard_normal((20_000, 10)); B = len(X)
m1p = np.stack([np.max(np.c_[np.zeros((B, 2)), X[:, :t]], 1) for t in range(10)], 1); rec = X > m1p
S = scores(net, X); P = S.argmax(1); o = X.argsort(1); sec, lar = o[:, -2], o[:, -1]
cons = (rec[:, :-1] & rec[:, 1:]).any(1)
ordn = ~rec[:, 8] & ~rec[:, 9] & ~cons & ((X > 0).sum(1) >= 2)
bad = np.flatnonzero(ordn & (P == lar))[:6]
for j in bad:
    print("x     ", X[j]); print("record", rec[j].astype(int), " second at", sec[j], "largest at", lar[j]); print("scores", S[j] - S[j].mean()); print()
# simplest controlled case: lone records, fillers -1
for vals in ({1: 0.5, 4: 1.0}, {2: 1.0, 5: 1.5}, {2: 1.0, 3: 1.5}, {0: 1.0, 6: 1.5}):
    x = np.full(10, -1.0)
    for k, v in vals.items(): x[k] = v
    s = scores(net, x[None])[0]; print(vals, s - s.mean())

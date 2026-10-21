"""Old largest at r, current largest at L. With each list's on/off pattern fixed, split the slope of every final score
(minus the mean) on the old largest by which neuron group carries it at step s (s = L-1, L, L+1)."""
import numpy as np
from algzoo.core import example
wi, wh, wo = (np.asarray(w, dtype=float) for w in example()); wi = wi.reshape(-1)
X = np.random.default_rng(45).standard_normal((600_000, 10)); B = len(X)
m1p = np.stack([np.max(np.c_[np.zeros((B, 1)), X[:, :t]], 1) for t in range(10)], 1); rec = X > m1p
recpos = [np.flatnonzero(q) for q in rec]
r = np.array([q[-2] if len(q) >= 2 else -1 for q in recpos]); L = np.array([q[-1] if len(q) else -1 for q in recpos])
groups = {"store": [2], "cancellers": [1, 6, 7, 8], "memory": [0, 3, 10, 11, 12, 13, 14, 15], "det,keep,late": [4, 5, 9]}
np.set_printoptions(precision=0, suppress=True, linewidth=170)
for rr, LL in ((1, 6), (2, 7), (1, 4), (3, 8)):
    m = (r == rr) & (L == LL); Xs = X[m]; n = len(Xs)
    h = np.zeros((n, 16)); G = np.zeros((n, 16)); ons = []; Gs = []
    for t in range(10):
        pre = h @ wh.T + Xs[:, t:t + 1] * wi; on = (pre > 0).astype(float); ons.append(on)
        G = G @ wh.T; G += wi * (t == rr); G *= on; h = np.maximum(pre, 0); Gs.append(G.copy())
    print(f"=== old at {rr}, current at {LL} ({n} lists); positions 0-9")
    for s in (LL - 1, LL, LL + 1):
        if s > 9: continue
        lam = np.broadcast_to(wo - wo.mean(0), (n, 10, 16)).copy()
        for t in range(9, s, -1): lam = np.einsum("npj,jk->npk", lam * ons[t][:, None, :], wh)
        print(f"  step {s}:")
        for g, idx in groups.items():
            print(f"    {g:14s}", (lam[:, :, idx] * Gs[s][:, None, idx]).sum(2).mean(0))

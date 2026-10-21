"""Where does the old largest's value, still held by the store when the current largest arrives (step L), end up in the
final scores? With each list's on/off pattern fixed: slope of every final score (minus the mean) on the old largest,
carried through the store at step L, versus carried through everything else at step L. Fixed (r, L) position pairs."""
import numpy as np
from algzoo.core import example
wi, wh, wo = (np.asarray(w, dtype=float) for w in example()); wi = wi.reshape(-1)
X = np.random.default_rng(45).standard_normal((600_000, 10)); B = len(X)
m1p = np.stack([np.max(np.c_[np.zeros((B, 1)), X[:, :t]], 1) for t in range(10)], 1); rec = X > m1p
recpos = [np.flatnonzero(q) for q in rec]
r = np.array([q[-2] if len(q) >= 2 else -1 for q in recpos]); L = np.array([q[-1] if len(q) else -1 for q in recpos])
np.set_printoptions(precision=1, suppress=True, linewidth=170)
for rr, LL in ((1, 4), (2, 5), (2, 7), (3, 6), (4, 7)):
    m = (r == rr) & (L == LL); Xs = X[m]; n = len(Xs)
    h = np.zeros((n, 16)); G = np.zeros((n, 16)); ons = []
    for t in range(10):
        pre = h @ wh.T + Xs[:, t:t + 1] * wi; on = (pre > 0).astype(float); ons.append(on)
        G = G @ wh.T; G[:, :] += wi * (t == rr); G *= on; h = np.maximum(pre, 0)
        if t == LL: GL = G.copy()
    lam = wo - wo.mean(0); lam = np.broadcast_to(lam, (n, 10, 16)).copy()
    for t in range(9, LL, -1): lam = np.einsum("npj,jk->npk", lam * ons[t][:, None, :], wh)
    via_store = (lam[:, :, 2] * GL[:, None, 2]).mean(0); total = np.einsum("npk,nk->np", lam, GL).mean(0)
    print(f"old at {rr}, current at {LL} ({n} lists)\n   via store at L: {via_store}\n   via all at L:   {total}")

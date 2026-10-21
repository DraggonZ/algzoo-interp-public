"""Three new largests at a < r < L (values P_prev, P, current). With each list's on/off pattern fixed, the slope of
every final score (minus the mean) on P_prev and on P, for fixed (a, r, L). If P_prev only shifts all candidate
positions (r and after L) equally, the old largest's jump code is harmless."""
import numpy as np
from algzoo.core import example
wi, wh, wo = (np.asarray(w, dtype=float) for w in example()); wi = wi.reshape(-1)
X = np.random.default_rng(46).standard_normal((1_500_000, 10)); B = len(X)
m1p = np.stack([np.max(np.c_[np.zeros((B, 1)), X[:, :t]], 1) for t in range(10)], 1); rec = X > m1p
recpos = [np.flatnonzero(q) for q in rec]
trip = np.array([tuple(q[-3:]) if len(q) >= 3 else (-1, -1, -1) for q in recpos])
np.set_printoptions(precision=0, suppress=True, linewidth=170)
for a, r, L in ((0, 2, 5), (0, 2, 6), (1, 3, 6), (0, 3, 7), (1, 4, 7)):
    m = (trip == (a, r, L)).all(1); Xs = X[m]; n = len(Xs)
    h = np.zeros((n, 16)); G = np.zeros((n, 16, 10))
    for t in range(10):
        pre = h @ wh.T + Xs[:, t:t + 1] * wi; on = (pre > 0).astype(float)
        G = np.einsum("ij,njt->nit", wh, G); G[:, :, t] += wi; G *= on[:, :, None]; h = np.maximum(pre, 0)
    J = np.einsum("pi,nit->npt", wo, G); J -= J.mean(1, keepdims=True)
    print(f"a={a} r={r} L={L} ({n} lists)\n   slope on P_prev: {J[:, :, a].mean(0)}\n   slope on P:      {J[:, :, r].mean(0)}\n   slope on current:{J[:, :, L].mean(0)}")

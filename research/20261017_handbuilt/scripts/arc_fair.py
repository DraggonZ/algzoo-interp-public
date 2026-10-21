"""Same lists as arc_margin_split (second = old largest P at r; rival y after the current largest L). Mean slopes of the
two candidates' scores (minus the mean score) on P, on y and on the current largest, i.e. where each input's push lands."""
import numpy as np
from algzoo.core import example
wi, wh, wo = (np.asarray(w, dtype=float) for w in example()); wi = wi.reshape(-1)
X = np.random.default_rng(42).standard_normal((400_000, 10)); B = len(X)
m1p = np.stack([np.max(np.c_[np.zeros((B, 1)), X[:, :t]], 1) for t in range(10)], 1); rec = X > m1p
o = X.argsort(1); lar, sec, thi = o[:, -1], o[:, -2], o[:, -3]; i = np.arange(B)
keep = rec[i, sec] & (thi > lar) & (X[i, thi] > 0) & (sec < lar)
X = X[keep]; lar, sec, thi = lar[keep], sec[keep], thi[keep]; n = len(X); i = np.arange(n)
h = np.zeros((n, 16)); G = np.zeros((n, 16, 10))
for t in range(10):
    pre = h @ wh.T + X[:, t:t + 1] * wi; on = (pre > 0).astype(float)
    G = np.einsum("ij,njt->nit", wh, G); G[:, :, t] += wi; G *= on[:, :, None]; h = np.maximum(pre, 0)
J = np.einsum("pi,nit->npt", wo, G); J -= J.mean(1, keepdims=True)
for name, src in (("old largest P", sec), ("rival y", thi), ("current largest", lar)):
    print(f"slope on {name:16s}: score at P {J[i, sec, src].mean():6.1f}   score at y {J[i, thi, src].mean():6.1f}")
print("by rival position (9 is read outside the memory):")
for p in range(4, 10):
    m = thi == p
    print(f"  y at {p} ({m.sum():5d}): on P -> at P {J[i, sec, sec][m].mean():5.1f}, at y {J[i, thi, sec][m].mean():5.1f};  "
          f"on y -> at y {J[i, thi, thi][m].mean():5.1f}, at P {J[i, sec, thi][m].mean():5.1f}")

"""Exact split of the trained network's decision margin by input. No biases, so inside each list's on/off pattern the
scores are s = J x exactly (J = slopes, measured by finite differences). Lists: the second is the old largest P (a new
largest that a later, bigger number overtook) and the third (rival y) comes after the current largest. The margin
s_P - s_y = sum_t (J[P,t] - J[y,t]) x_t is split by the role of input t. Coefficients are averaged per role, and the
share each role contributes to the margin is reported, in units of P."""
import numpy as np
from algzoo.core import example
wi, wh, wo = (np.asarray(w, dtype=float) for w in example()); wi = wi.reshape(-1)
def sc(X):
    h = np.zeros((len(X), 16))
    for t in range(10): h = np.maximum(h @ wh.T + X[:, t:t + 1] * wi, 0)
    return h @ wo.T
X = np.random.default_rng(42).standard_normal((400_000, 10)); B = len(X)
m1p = np.stack([np.max(np.c_[np.zeros((B, 1)), X[:, :t]], 1) for t in range(10)], 1); rec = X > m1p
o = X.argsort(1); lar, sec, thi = o[:, -1], o[:, -2], o[:, -3]; i = np.arange(B)
keep = rec[i, sec] & (thi > lar) & (X[i, thi] > 0) & (sec < lar)
X = X[keep]; rec = rec[keep]; lar, sec, thi = lar[keep], sec[keep], thi[keep]; n = len(X); i = np.arange(n)
s0 = sc(X); h = np.zeros((n, 16)); G = np.zeros((n, 16, 10))   # G = dh/dx with each list's on/off pattern fixed
for t in range(10):
    pre = h @ wh.T + X[:, t:t + 1] * wi; on = (pre > 0).astype(float)
    G = np.einsum("ij,njt->nit", wh, G); G[:, :, t] += wi[None, :]; G *= on[:, :, None]; h = np.maximum(pre, 0)
J = np.einsum("pi,nit->npt", wo, G)
assert np.allclose(np.einsum("npt,nt->np", J, X), s0, atol=1e-6)
coef = J[i, sec, :] - J[i, thi, :]                       # margin = sum_t coef_t x_t
P = X[i, sec]; contrib = coef * X / P[:, None]
pos = np.arange(10)[None, :]
roles = {"old largest P": pos == sec[:, None], "earlier records": rec & (pos < sec[:, None]),
         "current largest": pos == lar[:, None], "rival y": pos == thi[:, None],
         "other positives": (X > 0) & ~rec & (pos != thi[:, None]) & (pos != sec[:, None]),
         "negative numbers": X < 0}
wrong = (s0.argmax(1) != sec)
print(f"{n} lists; network wrong on {wrong.mean():.3f}; mean margin / P {((s0[i, sec] - s0[i, thi]) / P).mean():.1f}")
print(f"{'role':18s} {'mean coef':>9s} {'share of margin (mean, x P)':>28s}  {'when wrong':>10s}")
for k, m in roles.items():
    c = np.where(m, coef, 0).sum(1) / np.maximum(m.sum(1), 1); sh = np.where(m, contrib, 0).sum(1)
    print(f"{k:18s} {c[m.any(1)].mean():9.1f} {sh.mean():28.1f}  {sh[wrong].mean():10.1f}")
print("old largest coefficient vs rival coefficient by gap between their positions (rival - P):")
for g in range(2, 10):
    m = (thi - sec) == g
    if m.sum() > 300:
        print(f"  gap {g}: lists {m.sum():6d}  coef on P {coef[i, sec][m].mean():6.1f}  coef on y {coef[i, thi][m].mean():6.1f}  wrong {wrong[m].mean():.3f}")

print("\nper-list spread of the coefficients (std): P %.1f, rival %.1f, earlier records %.1f, current largest %.1f" % (
    coef[i, sec].std(), coef[i, thi].std(), coef[rec & (pos < sec[:, None])].std(), coef[i, lar].std()))
groups = {"cancellers 1,6,7,8": [1, 6, 7, 8], "memory 0,3,10-15": [0, 3, 10, 11, 12, 13, 14, 15], "store 2": [2],
          "detector 4": [4], "keep 5": [5], "late 9": [9]}
dw = wo[sec] - wo[thi]                                   # (n, 16): readout difference between the two candidates
print("margin coefficient split by the final neurons that carry it (mean over lists)")
print(f"{'input role':18s} " + " ".join(f"{g[:14]:>14s}" for g in groups))
for k in ("old largest P", "earlier records", "current largest", "rival y"):
    m = roles[k]; row = []
    for g, idx in groups.items():
        part = np.einsum("ni,nit->nt", dw[:, idx], G[:, idx, :])     # contribution of this group to coef_t
        row.append((np.where(m, part, 0).sum(1) / np.maximum(m.sum(1), 1))[m.any(1)].mean())
    print(f"{k:18s} " + " ".join(f"{v:14.1f}" for v in row))

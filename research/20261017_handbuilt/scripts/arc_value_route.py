"""How does the old largest's value reach the trained network's decision? Same lists as arc_margin_split (second =
old largest P at r, rival y after the current largest). With each list's on/off pattern fixed, the margin's slope on
x_r splits exactly over the neurons at any one step t >= r: sum_k lambda_t[k] * dh_t[k]/dx_r, where lambda_t is the
margin's slope on h_t. Report that split at t = r, r+1, r+2, r+3 (mean over lists), for x_r and, as a contrast, for
the earlier record P_prev and for the current largest."""
import numpy as np
from algzoo.core import example
wi, wh, wo = (np.asarray(w, dtype=float) for w in example()); wi = wi.reshape(-1)
X = np.random.default_rng(42).standard_normal((400_000, 10)); B = len(X)
m1p = np.stack([np.max(np.c_[np.zeros((B, 1)), X[:, :t]], 1) for t in range(10)], 1); rec = X > m1p
o = X.argsort(1); lar, sec, thi = o[:, -1], o[:, -2], o[:, -3]; i = np.arange(B)
keep = rec[i, sec] & (thi > lar) & (X[i, thi] > 0) & (sec < lar) & (lar - sec >= 3)
X = X[keep]; rec = rec[keep]; lar, sec, thi = lar[keep], sec[keep], thi[keep]; n = len(X); i = np.arange(n)
h = np.zeros((n, 16)); G = np.zeros((n, 16, 10)); Gs, ons = [], []
for t in range(10):
    pre = h @ wh.T + X[:, t:t + 1] * wi; on = (pre > 0).astype(float)
    G = np.einsum("ij,njt->nit", wh, G); G[:, :, t] += wi[None, :]; G *= on[:, :, None]; h = np.maximum(pre, 0)
    Gs.append(G.copy()); ons.append(on)
lam = [None] * 10; lam[9] = wo[sec] - wo[thi]
for t in range(8, -1, -1): lam[t] = (lam[t + 1] * ons[t + 1]) @ wh
recpos = [np.flatnonzero(r) for r in rec]
prev = np.array([rp[rp < s][-1] if (rp < s).any() else -1 for rp, s in zip(recpos, sec)])
names = {0: "m0", 1: "c1", 2: "store", 3: "m3", 4: "det", 5: "keep", 6: "c6", 7: "c7", 8: "c8", 9: "late",
         10: "m10", 11: "m11", 12: "m12", 13: "m13", 14: "m14", 15: "m15"}
np.set_printoptions(precision=1, suppress=True, linewidth=200)
for label, src, m in (("old largest x_r", sec, np.ones(n, bool)), ("earlier record P_prev", prev, prev >= 0),
                      ("current largest", lar, lar <= 6)):
    print(f"--- {label} ({m.sum()} lists): margin slope carried by each neuron k at step src + d")
    print("      " + " ".join(f"{names[k]:>6s}" for k in range(16)) + "    total")
    for d in range(4):
        t = src + d; ok = m & (t <= 9)
        part = np.stack([lam[tt][j] * Gs[tt][j, :, s] for j, tt, s in zip(np.flatnonzero(ok), t[ok], src[ok])])
        print(f"  +{d}  " + " ".join(f"{v:6.1f}" for v in part.mean(0)) + f"   {part.sum(1).mean():6.1f}")

print("\n--- old largest x_r, split at the steps around the current largest's arrival (step L = position of the current largest)")
print("      " + " ".join(f"{names[k]:>6s}" for k in range(16)) + "    total")
for d in (-1, 0, 1, 2, 3):
    t = lar + d; ok = (t <= 9) & (t >= sec)
    part = np.stack([lam[tt][j] * Gs[tt][j, :, s] for j, tt, s in zip(np.flatnonzero(ok), t[ok], sec[ok])])
    print(f"  L{d:+d}  " + " ".join(f"{v:6.1f}" for v in part.mean(0)) + f"   {part.sum(1).mean():6.1f}")

J = np.einsum("pi,nit->npt", wo, Gs[9]); Jc = J - J.mean(1, keepdims=True)
print("\nslope of each final score (minus the mean score) on the old largest x_r, for fixed positions of the old (r) and current (L) largest")
for r, L in ((1, 4), (1, 6), (2, 5), (2, 7), (3, 5), (3, 8), (4, 7)):
    m = (sec == r) & (lar == L)
    if m.sum() > 100:
        print(f"  r={r} L={L} ({m.sum():5d} lists): " + " ".join(f"{v:6.1f}" for v in Jc[m][:, :, None][np.arange(m.sum()), :, 0][:, :, None].mean(0)[:, 0]) if False else
              f"  r={r} L={L} ({m.sum():5d} lists): " + " ".join(f"{v:6.1f}" for v in Jc[m, :, r].mean(0)) +
              "   | on the current largest: " + " ".join(f"{v:6.1f}" for v in Jc[m, :, L].mean(0)))

print("\n--- rival y (kept number after the current largest): route at steps y, y+1, y+2")
print("      " + " ".join(f"{names[k]:>6s}" for k in range(16)) + "    total")
for d in range(3):
    t = thi + d; ok = t <= 9
    part = np.stack([lam[tt][j] * Gs[tt][j, :, s] for j, tt, s in zip(np.flatnonzero(ok), t[ok], thi[ok])])
    print(f"  +{d}  " + " ".join(f"{v:6.1f}" for v in part.mean(0)) + f"   {part.sum(1).mean():6.1f}   ({ok.sum()} lists)")

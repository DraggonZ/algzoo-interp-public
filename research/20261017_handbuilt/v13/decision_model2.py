"""Like decision_model.py, one-sided bump with free shape: 1 at r, `prev` at r-1, `lobe` at r-3 and below (linear
ramp from 0 at r-2), nothing after r. Grid over F, kappa, prev, lobe; c = 0.75."""
import numpy as np, itertools
X = np.random.default_rng(50).standard_normal((200_000, 10)); B = len(X)
srt = [np.sort(np.c_[np.zeros((B, 2)), X[:, :t]], 1) for t in range(10)]
m2 = np.stack([q[:, -2] for q in srt], 1); m1 = np.stack([q[:, -1] for q in srt], 1)
rec = X > m1; lar = X.argmax(1); o = X.argsort(1); sec, thi = o[:, -2], o[:, -3]; onepos = (X > 0).sum(1) <= 1
pos = np.arange(10); c = 0.75
v = np.maximum(0, (1 - c) * X - c * np.maximum(m2 - X, 0)) * (~rec)
def run(F, kappa, prev, lobe):
    S = v.copy(); peak = np.zeros_like(X)
    for r in range(10):
        m = rec[:, r] & (lar != r); d = r - pos
        shp = np.where(d == 0, 1.0, np.where(d == 1, prev, np.where(d >= 3, lobe * (d - 2), 0.0)))
        S[m] += F * (1 - c) * X[m, r:r + 1] * shp[None, :]; peak[m, r] = F * (1 - c) * X[m, r]
    w = v + peak; later = np.cumsum(w[:, ::-1], 1)[:, ::-1] - w
    S = S - kappa * later; P = S.argmax(1); ok = P == sec; bad = ~ok
    return ok.mean(), (P == thi)[bad].mean(), (P == lar)[bad].mean(), onepos[bad].mean()
res = []
for F, kappa, prev, lobe in itertools.product((1, 1.5, 2, 3), (0, 0.3, 0.6, 1, 2), (0, 0.35, 0.7), (0, -0.35, -0.7)):
    res.append((run(F, kappa, prev, lobe), F, kappa, prev, lobe))
res.sort(key=lambda z: -z[0][0])
print("best 12 (accuracy, mistakes third/largest/onepos): F kappa prev lobe")
for (a, t, l, op), F, k, p, lb in res[:12]: print(f"  {a:.3f}  {t:.2f}/{l:.2f}/{op:.2f}   F {F} kappa {k} prev {p} lobe {lb}")
print("with ARC-like prev 0.7:")
for (a, t, l, op), F, k, p, lb in [z for z in res if z[3] == 0.7][:5]: print(f"  {a:.3f}  {t:.2f}/{l:.2f}/{op:.2f}   F {F} kappa {k} prev {p} lobe {lb}")

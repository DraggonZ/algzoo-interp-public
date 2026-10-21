"""Leaky-hiding target code at the level of decisions (no neurons). Per list:
- positive non-record x at p: relative vote max(0, (1-c) x - c (m2 - x)+);
- every record j at r_j (including the current largest): a one-sided bump (1-c) F x_j b(p - r_j), b(0)=1, b(-1)=s;
- the clock hides only the current largest, with a symmetric wave H (1-c) (x_n - alpha P_old) cos(45 deg (p - r_n)),
  where P_old is the largest before it (alpha = share driven by the jump);
- optional push-down kappa of earlier positions by every vote and bump peak.
Accuracy and mistakes (third / largest / at most one positive)."""
import numpy as np, itertools
X = np.random.default_rng(51).standard_normal((200_000, 10)); B = len(X)
srt = [np.sort(np.c_[np.zeros((B, 2)), X[:, :t]], 1) for t in range(10)]
m1 = np.stack([q[:, -1] for q in srt], 1); m2 = np.stack([q[:, -2] for q in srt], 1)
rec = X > m1; lar = X.argmax(1); o = X.argsort(1); sec, thi = o[:, -2], o[:, -3]; onepos = (X > 0).sum(1) <= 1
pos = np.arange(10); c = 0.75; i = np.arange(B)
v = np.maximum(0, (1 - c) * X - c * np.maximum(m2 - X, 0)) * (~rec)
Pold = m1[i, lar]
def run(F, s, H, alpha, kappa):
    S = v.copy(); peak = np.zeros_like(X)
    for r in range(10):
        m = rec[:, r]; d = pos - r
        b = np.where(d == 0, 1.0, np.where(d == -1, s, 0.0))
        S[m] += F * (1 - c) * X[m, r:r + 1] * b[None, :]; peak[m, r] = F * (1 - c) * X[m, r]
    d = pos[None, :] - lar[:, None]
    S -= H * (1 - c) * (X[i, lar] - alpha * Pold)[:, None] * np.cos(np.pi / 4 * d)
    w = v + peak * (pos[None, :] != lar[:, None]); later = np.cumsum(w[:, ::-1], 1)[:, ::-1] - w
    S -= kappa * later
    P = S.argmax(1); ok = P == sec; bad = ~ok
    return ok.mean(), (P == thi)[bad].mean(), (P == lar)[bad].mean(), onepos[bad].mean()
res = []
for F, s, H, alpha, kappa in itertools.product((1, 2, 3), (0, 0.35), (0.5, 1, 1.5, 2, 3), (0, 0.7, 1), (0, 0.5)):
    res.append((run(F, s, H, alpha, kappa), (F, s, H, alpha, kappa)))
res.sort(key=lambda z: -z[0][0])
print("best 15: accuracy, mistakes third/largest/onepos | F, spill, H, jump share alpha, kappa")
for (a, t, l, op), prm in res[:15]: print(f"  {a:.3f}  {t:.2f}/{l:.2f}/{op:.2f}  {prm}")
print("alpha = 0.7 (trained network's jump share), best 5:")
for (a, t, l, op), prm in [z for z in res if z[1][3] == 0.7][:5]: print(f"  {a:.3f}  {t:.2f}/{l:.2f}/{op:.2f}  {prm}")

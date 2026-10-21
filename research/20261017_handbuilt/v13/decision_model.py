"""v13 target code at the level of decisions (no neurons). Scores per list:
- every positive non-record x at p: vote v = max(0, (1-c) x - c (m2 - x)+), m2 = second-largest before p;
- every overtaken record (new largest when it arrived, not the final largest) at r: a bump F (1-c) x_r shape(p - r),
  shape = cosine at 45 deg per step, either two-sided or one-sided (only p <= r, plus `after` x the cosine for p > r);
- the final largest: nothing (hidden);
- push-down: every vote and every bump peak lowers all earlier positions by kappa x its size.
Reports accuracy and the mistake mix (third / largest / at most one positive)."""
import numpy as np
X = np.random.default_rng(50).standard_normal((300_000, 10)); B = len(X)
srt = [np.sort(np.c_[np.zeros((B, 2)), X[:, :t]], 1) for t in range(10)]
m1 = np.stack([q[:, -1] for q in srt], 1); m2 = np.stack([q[:, -2] for q in srt], 1)
rec = X > m1; lar = X.argmax(1); o = X.argsort(1); sec, thi = o[:, -2], o[:, -3]; onepos = (X > 0).sum(1) <= 1
pos = np.arange(10)
def run(c=0.75, F=1.0, kappa=0.0, after=1.0):
    v = np.maximum(0, (1 - c) * X - c * np.maximum(m2 - X, 0)) * (~rec)
    S = v.copy()
    peak = np.zeros_like(X)
    for r in range(10):
        m = rec[:, r] & (lar != r)
        d = pos[None, :] - r
        shp = np.cos(np.pi / 4 * d) * np.where(d > 0, after, 1.0)
        S[m] += F * (1 - c) * X[m, r:r + 1] * shp[0]
        peak[m, r] = F * (1 - c) * X[m, r]
    w = v + peak
    later = np.cumsum(w[:, ::-1], 1)[:, ::-1] - w            # sum of votes at later positions
    S = S - kappa * later
    P = S.argmax(1); ok = P == sec; bad = ~ok
    return ok.mean(), (P == thi)[bad].mean(), (P == lar)[bad].mean(), onepos[bad].mean()
print("trained network: 0.953, mistakes third 0.61 largest 0.15 one-positive 0.18")
for c in (0.75, 0.5):
    for after in (1.0, 0.0):
        for F in (1.0, 2.0, 3.0):
            for kappa in (0.0, 0.3, F - 1):
                a, t, l, op = run(c, F, kappa, after)
                print(f"c {c} {'two' if after else 'one'}-sided F {F} kappa {kappa:.1f}: acc {a:.3f}  mistakes third {t:.2f} largest {l:.2f} onepos {op:.2f}")

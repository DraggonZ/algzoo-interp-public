"""deduce_width.py predicted the zone near the tie well but its tail too wide (the map made at a near tie is wrong for
large gaps, where other neurons switch). Fix, decided before running: one canonical list per arrangement and per gap
value (third / second = 0.995, 0.97, 0.93, 0.85, 0.7, 0.5), each list using the map of its nearest gap."""
import sys; sys.path.insert(0, "../steer"); sys.path.insert(0, "../level2/deduce")
import numpy as np
from scipy.stats import norm
from harness import *
from case_maps import A, idx, C
R = [0.995, 0.97, 0.93, 0.85, 0.7, 0.5]; Js = []
for r0 in R:
    C2 = C.clone(); mu_r = -norm.pdf(r0) / norm.cdf(r0)
    for i, (l, p, q) in enumerate(A): C2[i] = mu_r; C2[i, l], C2[i, p], C2[i, q] = 1.539, 1.0, r0
    Js.append(jac(ARC, C2).cpu().numpy())
Js = np.stack(Js)                                               # gaps x 720 x 10 x 10
X = lists(4_000_000, seed=502); o = X.argsort(1, descending=True); v = X.sort(1, descending=True).values
m = v[:, 1] > 0.1; Xn = X[m].cpu().numpy(); on = o[m].cpu().numpy(); vn = v[m].cpu().numpy()
a_true = (run(ARC, X[m]).argmax(1) == o[m][:, 2]).cpu().numpy()
ai = np.array([idx[(l, s, t)] for l, s, t in on[:, :3]]); ratio = vn[:, 2] / vn[:, 1]
gi = np.abs(ratio[:, None] - np.array(R)[None]).argmin(1)
pred = np.zeros(len(Xn))
for k in np.unique(ai):
    l, s, t = A[k]; others = [j for j in range(10) if j not in (l, s, t)]
    for gk in range(len(R)):
        sel = np.where((ai == k) & (gi == gk))[0]
        if not len(sel): continue
        c = Js[gk, k, s] - Js[gk, k, t]; x3 = vn[sel, 2]
        mtr = -norm.pdf(x3) / norm.cdf(x3); vtr = 1 - x3 * norm.pdf(x3) / norm.cdf(x3) - mtr ** 2
        mean = c[s] * Xn[sel, s] + c[t] * Xn[sel, t] + c[l] * Xn[sel, l] + c[others].sum() * mtr
        sd = np.sqrt((c[others] ** 2).sum() * np.maximum(vtr, 1e-9)); pred[sel] = norm.cdf(-mean / sd)
g = (vn[:, 1] - vn[:, 2]) / vn[:, 1]
bins = [0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.2, 2]
print("third wins, observed / predicted, by gap:")
print("   " + " ".join(f"{lo:.3f}-{hi:.3f}: {100*a_true[(g >= lo) & (g < hi)].mean():.1f}/{100*pred[(g >= lo) & (g < hi)].mean():.1f}" for lo, hi in zip(bins, bins[1:])))
print(f"share where the third wins: observed {100*a_true.mean():.2f}%, predicted {100*pred.mean():.2f}% (cost about {len(R)*720*11:,} run-equivalents)")

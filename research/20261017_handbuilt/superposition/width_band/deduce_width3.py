"""Cheaper version of the exact one-number-at-a-time treatment. Per arrangement and per gap value (third/second = 0.99,
0.93, 0.7), on the canonical list: run the network with each smaller number alone moved over three quadrature nodes of
its law (normal truncated below the third; others at their mean), so each number's effect on the lead includes the
switching. Mean lead = canonical lead + sum of the numbers' mean shifts + the top three's deviation through the linear
map; spread = square root of the sum of their variances; P(third wins) = Phi(-mean / spread). No network run on random
lists. Prediction (written first): the tail comes down toward the truth (the linear version overstated it)."""
import sys; sys.path.insert(0, "../steer"); sys.path.insert(0, "../level2/deduce")
import numpy as np
from scipy.stats import norm
from scipy.special import roots_hermitenorm
from harness import *
from case_maps import A, idx, C
R = [0.99, 0.93, 0.7]; Q = 3
def trunc_nodes(b):
    # three-point rule for N(0,1) truncated below b: quantile midpoints of three equal-probability bins
    p = norm.cdf(b); qs = np.array([1/6, 1/2, 5/6]) * p; return norm.ppf(qs)
lead0, shift, var, Jt = {}, {}, {}, {}
runs = 0
for gk, r0 in enumerate(R):
    mu_r = -norm.pdf(r0) / norm.cdf(r0); base = C.clone()
    for i, (l, p, q) in enumerate(A): base[i] = mu_r; base[i, l], base[i, p], base[i, q] = 1.539, 1.0, r0
    J = jac(ARC, base).cpu().numpy(); S0 = scores(ARC, base); runs += 720 * 11
    nodes = trunc_nodes(r0)
    Xs = base.repeat_interleave(10 * Q, 0).clone(); jj = torch.arange(10, device=DEV).repeat_interleave(Q).repeat(720); kk = torch.tensor(np.tile(nodes, 10 * 720), device=DEV, dtype=dt)
    Xs[torch.arange(len(Xs), device=DEV), jj] = kk; Sx = scores(ARC, Xs).reshape(720, 10, Q, 10); runs += len(Xs)
    for i, (l, p, q) in enumerate(A):
        ld = (Sx[i, :, :, p] - Sx[i, :, :, q]).cpu().numpy(); l0 = (S0[i, p] - S0[i, q]).item()
        others = [j for j in range(10) if j not in (l, p, q)]
        lead0[gk, i] = l0; shift[gk, i] = sum(ld[j].mean() - l0 for j in others); var[gk, i] = sum(ld[j].var() for j in others)
        Jt[gk, i] = (J[i, p] - J[i, q])
X = lists(4_000_000, seed=502); o = X.argsort(1, descending=True); v = X.sort(1, descending=True).values
m = v[:, 1] > 0.1; Xn = X[m].cpu().numpy(); on = o[m].cpu().numpy(); vn = v[m].cpu().numpy()
a_true = (run(ARC, X[m]).argmax(1) == o[m][:, 2]).cpu().numpy()
ai = np.array([idx[(l, s, t)] for l, s, t in on[:, :3]]); ratio = vn[:, 2] / vn[:, 1]; gi = np.abs(ratio[:, None] - np.array(R)[None]).argmin(1)
pred = np.zeros(len(Xn))
for k in np.unique(ai):
    l, s, t = A[k]
    for gk in range(len(R)):
        sel = np.where((ai == k) & (gi == gk))[0]
        if not len(sel): continue
        sc = vn[sel, 1]                                   # scale: everything is linear in the overall size
        dtop = Jt[gk, k][l] * (Xn[sel, l] / sc - 1.539) + Jt[gk, k][s] * (Xn[sel, s] / sc - 1.0) + Jt[gk, k][t] * (Xn[sel, t] / sc - R[gk])
        mean = lead0[gk, k] + shift[gk, k] + dtop; sd = np.sqrt(max(var[gk, k], 1e-12))
        pred[sel] = norm.cdf(-mean / sd)
g = (vn[:, 1] - vn[:, 2]) / vn[:, 1]; bins = [0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.2, 2]
print("third wins, observed / predicted, by gap:")
print("   " + " ".join(f"{lo:.3f}-{hi:.3f}: {100*a_true[(g >= lo) & (g < hi)].mean():.1f}/{100*pred[(g >= lo) & (g < hi)].mean():.1f}" for lo, hi in zip(bins, bins[1:])))
print(f"share where the third wins: observed {100*a_true.mean():.2f}%, predicted {100*pred.mean():.2f}% (cost {runs:,} run-equivalents; sampling at that cost: about ±{100*np.sqrt(0.0286*0.97/runs):.2f} points)")

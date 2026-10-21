"""Near ties (second minus third < 5% of the second, second > 0): how often the third wins, from the weights only,
by integrating directly over that thin region (the main grid does not resolve its edge). m2 on Gauss-Legendre (0, 4),
a = 0.05 m2 t with t on Gauss-Legendre (0, 1), b = 5 s^2. Same per-point models as estimate.py (frozen 'final')."""
import numpy as np, torch
from scipy.stats import norm
from lib import *
import estimate as E
W = load_net("ARC"); n = 16
x, w = np.polynomial.legendre.leggauss(n); m2 = 2 * (x + 1); w2 = 2 * w
xt, wt_ = np.polynomial.legendre.leggauss(8); t = 0.5 * (xt + 1); wt_ = 0.5 * wt_
s = 0.5 * (x + 1); b = 5 * s ** 2; wb = 5 * 2 * s * 0.5 * w
M2, T_, B = np.meshgrid(m2, t, b, indexing="ij"); W2, WT, WB = np.meshgrid(w2, wt_, wb, indexing="ij")
M2, T_, B, W2, WT, WB = (q.ravel() for q in (M2, T_, B, W2, WT, WB))
a = 0.05 * M2 * T_; m3 = M2 - a; m1 = M2 + B
wgt = W2 * WT * WB * 0.05 * M2 * norm.pdf(m1) * norm.pdf(M2) * norm.pdf(m3) * norm.cdf(m3) ** 7
V = torch.tensor(np.c_[m1, M2, m3], device=DEV, dtype=dt); wg = torch.tensor(wgt, device=DEV, dtype=dt)
perms = [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]; names = ["L,2,3", "L,3,2", "2,L,3", "3,L,2", "2,3,L", "3,2,L"]
rk = TRIPLES.argsort(1).argsort(1); oid = torch.zeros(len(TRIPLES), dtype=torch.long, device=DEV)
for i, p in enumerate(perms): oid[(rk == torch.tensor(p, device=DEV)).all(1)] = i
z = np.load("truth_stats.npz"); nt = z["nt"]
print(f"near-tie share of lists: weights only {100*720*wg.sum().item():.2f}%, sampled {100*nt[:,1].sum()/z['N']:.2f}%")
print(f"truth (sampled): third wins {100*nt[:,0].sum()/nt[:,1].sum():.1f}%; by order " + "  ".join(f"{names[i]} {100*nt[i,0]/nt[i,1]:.1f}" for i in range(6)))
for model, q in [("mean01", 0), ("two", 0), ("final", 4), ("v2", 4)]:
    num = torch.zeros(6, device=DEV, dtype=dt); den = torch.zeros(6, device=DEV, dtype=dt)
    for j in range(0, len(TRIPLES), 24):
        tt = TRIPLES[j:j + 24]; pos = tt.repeat_interleave(len(V), 0); val = V.repeat(len(tt), 1); ww = wg.repeat(len(tt))
        o = oid[j:j + 24].repeat_interleave(len(V))
        for c in range(0, len(pos), 6000):
            sl = slice(c, c + 6000); ok, sp, _ = E.per_point(W, pos[sl], val[sl], model, q)
            num.index_add_(0, o[sl], sp[:, 1] * ww[sl]); den.index_add_(0, o[sl], ww[sl])
    r = (num / den).cpu().numpy()
    print(f"{model:7s}: third wins {100*num.sum().item()/den.sum().item():.1f}%; by order " + "  ".join(f"{names[i]} {100*r[i]:.1f}" for i in range(6)), flush=True)

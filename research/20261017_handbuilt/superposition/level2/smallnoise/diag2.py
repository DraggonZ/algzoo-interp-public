"""Why is even the exact-moment Gaussian biased low? Oracle variants (all from sampled backgrounds; diagnosis only):
  oracle Gaussian          : one Gaussian for the nine leads (exact mean and covariance)
  oracle | 4th             : Gaussian within groups of backgrounds sharing the 4th largest's position and value bin
                             (8 equal-mass bins), averaged over groups -> is the non-Gaussianity the 4th largest?
  oracle | 4th, jac rest   : same groups, but the Gaussian from the exact local linear map at (4th exact, the other six at
                             their mean truncated below the 4th) -> can a weights-only calculation give it?
  oracle | 4th, 0/1 rest   : same groups, the other six at their mean, 0/1 answer
Also: the skewness of the most dangerous lead and which competitor it is (a background position or the largest/third)."""
import sys, numpy as np, torch
from lib import *
from diag import conditions, W
K = 4000; NB = 8


def run_one(pos_i, val_i):
    p = ncdf(val_i[2]).item(); U = torch.rand(K, 10, device=DEV, dtype=dt) * p; X = nicdf(U.clamp_min(1e-300))
    X[:, pos_i] = val_i; return X


def group_stats(pos, val):
    n = len(pos); out = {k: np.zeros(n) for k in ["truth", "oracle", "oracle|4th", "jac|4th", "mean|4th"]}
    A, others = lead_matrix(pos[:, 1]); kind = np.zeros(n, int); skew = np.zeros(n)
    for i in range(n):
        X = run_one(pos[i], val[i]); L = run(W, X) @ A[i].T; win = (L > 0).all(1).double()
        out["truth"][i] = win.mean().item()
        out["oracle"][i] = orthant(L.mean(0)[None], torch.cov(L.T)[None]).item()
        Xb = X.clone(); Xb[:, pos[i]] = -1e9; x4, p4 = Xb.max(1)
        q = (ncdf(x4) / ncdf(val[i, 2])).clamp(0, 1 - 1e-12); b = (q ** 7 * NB).long().clamp(max=NB - 1)  # x4's cdf is q^7
        g = p4 * NB + b
        acc_o = acc_j = acc_m = 0.0
        ug = torch.unique(g)
        # moments per group (oracle), and weights-only per group at the group's median x4
        lms, lcs, Xc, wts = [], [], [], []
        for gg in ug:
            m = g == gg; Lg = L[m]
            if m.sum() < 3: lms.append(Lg.mean(0)); lcs.append(torch.eye(9, device=DEV, dtype=dt) * 1e-6)
            else: lms.append(Lg.mean(0)); lcs.append(torch.cov(Lg.T))
            xm = x4[m].median(); xc = torch.full((10,), background_mean(xm[None])[0].item(), device=DEV, dtype=dt)
            xc[pos[i]] = val[i]; xc[p4[m][0]] = xm; Xc.append(xc); wts.append(m.double().mean())
        wts = torch.stack(wts); Xc = torch.stack(Xc)
        out["oracle|4th"][i] = (wts * orthant(torch.stack(lms), torch.stack(lcs))).sum().item()
        S, G = run_jac(W, Xc); isbg = torch.ones(len(Xc), 10, dtype=torch.bool, device=DEV); isbg[:, pos[i]] = False
        isbg[torch.arange(len(Xc)), p4[torch.stack([torch.nonzero(g == gg)[0, 0] for gg in ug])]] = False
        G = G * isbg[:, :, None]; Ai = A[i][None].expand(len(Xc), 9, 10)
        x4m = torch.tensor([x4[g == gg].median().item() for gg in ug], device=DEV, dtype=dt)
        nodes, w = bg_nodes(x4m, 32); mu = (nodes * w).sum(1); var = (nodes ** 2 * w).sum(1) - mu ** 2
        GL = G @ Ai.transpose(1, 2); lm = (Ai @ S[:, :, None])[..., 0]; lc = var[:, None, None] * GL.transpose(1, 2) @ GL
        out["jac|4th"][i] = (wts * orthant(lm, lc)).sum().item()
        out["mean|4th"][i] = (wts * (lm > 0).all(1).double()).sum().item()
        # most dangerous lead under the single Gaussian
        z = L.mean(0) / L.std(0); j = z.argmin().item(); k = others[i, j].item()
        kind[i] = 0 if k == pos[i, 0] else 1 if k == pos[i, 2] else 2
        d = L[:, j] - L[:, j].mean(); skew[i] = ((d ** 3).mean() / d.pow(2).mean() ** 1.5).item()
    return out, kind, skew


if __name__ == "__main__":
    torch.manual_seed(1)
    for label, near in [("random conditions", False), ("near-tie conditions", True)]:
        o, v = conditions(300, near); pos = torch.tensor(o, device=DEV); val = torch.tensor(v, device=DEV, dtype=dt)
        out, kind, skew = group_stats(pos, val); tr = out["truth"]
        print(f"{label}: truth mean P(right) {tr.mean():.4f}")
        for k in ["oracle", "oracle|4th", "jac|4th", "mean|4th"]:
            r = out[k]; print(f"  {k:12s} mean {r.mean():.4f}  bias {r.mean()-tr.mean():+.4f}  mean abs error {np.abs(r-tr).mean():.4f}  corr {np.corrcoef(r, tr)[0,1]:.3f}")
        for c, nm in enumerate(["the largest", "the third", "a smaller number"]):
            m = kind == c
            if m.any(): print(f"  most dangerous competitor is {nm}: {m.mean():.2f} of conditions; median skew of that lead {np.median(skew[m]):+.2f}; "
                              f"oracle bias there {(out['oracle'][m]-tr[m]).mean():+.4f}")

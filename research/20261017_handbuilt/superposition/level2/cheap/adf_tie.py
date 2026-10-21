"""Most favourable propagation test: positions and values of the three largest fixed (L, and the second and third tied
at u); the seven smaller numbers random (normal truncated below u). Push one Gaussian through the network (exact ReLU
mean and variance per neuron, covariance by the gain approximation, as in prop4) and compare the lean at the tie
(score of the second's position minus the third's) and P(lean < 0) with Monte Carlo over the smaller numbers."""
import sys, numpy as np, torch
sys.path.insert(0, "../../../family")
from common import ARC, trained, scores
from linelib import DEV, dt
Phi = torch.special.ndtr
def relu_m(m, v):
    s = v.clamp_min(1e-30).sqrt(); r = m / s; P = Phi(r); p = torch.exp(-0.5 * r * r) * 0.3989422804014327
    mean = m * P + s * p; var = ((m * m + v) * P + m * s * p - mean ** 2).clamp_min(0); return mean, var, P
def adf(W, xm, xv):
    wi, wh, wo = W; m = torch.zeros(16, device=DEV, dtype=dt); C = torch.zeros(16, 16, device=DEV, dtype=dt)
    for t in range(10):
        um = wh @ m + xm[t] * wi; U = wh @ C @ wh.T + xv[t] * torch.outer(wi, wi)
        mean, var, g = relu_m(um, torch.diagonal(U)); C = g[:, None] * g[None] * U; C[range(16), range(16)] = var; m = mean
    return wo @ m, wo @ C @ wo.T
rng = np.random.default_rng(7); out = []
for name, W in (("ARC", ARC), ("114", trained(114))):
    g = torch.Generator(device=DEV).manual_seed(0); errs = []
    print(name)
    for trial in range(12):
        pL, pS, pT = rng.choice(10, 3, replace=False); L = float(rng.uniform(1.2, 2.2)); u = float(rng.uniform(0.4, 1.2))
        rest = [i for i in range(10) if i not in (pL, pS, pT)]
        a = Phi(torch.tensor(u, dtype=dt)).item(); zm = -np.exp(-u * u / 2) / np.sqrt(2 * np.pi) / a; zv = 1 - u * (-zm) - zm ** 2
        xm = torch.zeros(10, device=DEV, dtype=dt); xv = torch.zeros(10, device=DEV, dtype=dt)
        xm[rest] = zm; xv[rest] = zv; xm[pL] = L; xm[pS] = u; xm[pT] = u
        sm, sC = adf(W, xm, xv); lm = (sm[pS] - sm[pT]).item(); lv = (sC[pS, pS] + sC[pT, pT] - 2 * sC[pS, pT]).item()
        n = 20000; Z = torch.special.ndtri(torch.rand(n, 7, device=DEV, dtype=dt, generator=g) * a)
        X = xm.repeat(n, 1); X[:, rest] = Z; s = scores(W, X); lam = s[:, pS] - s[:, pT]
        p_adf = Phi(torch.tensor(-lm / np.sqrt(max(lv, 1e-12)), dtype=dt)).item(); p_mc = (lam < 0).double().mean().item()
        hinge_adf = np.sqrt(lv) * (np.exp(-0.5 * (lm / np.sqrt(lv)) ** 2) / np.sqrt(2 * np.pi)) - lm * p_adf
        hinge_mc = (-lam).clamp_min(0).mean().item()
        print(f"  L@{pL}={L:.2f} S@{pS} T@{pT} u={u:.2f}: lean MC {lam.mean():+.2f} sd {lam.std():.2f} | one Gaussian {lm:+.2f} sd {np.sqrt(lv):.2f} | P(lean<0) MC {p_mc:.2f} Gaussian {p_adf:.2f} | E[max(0,-lean)] MC {hinge_mc:.2f} Gaussian {hinge_adf:.2f}")
        errs.append((hinge_mc, hinge_adf))
    e = np.array(errs); print(f"  E[max(0,-lean)] summed over configs: MC {e[:,0].sum():.2f}, one Gaussian {e[:,1].sum():.2f}; mean abs relative error {np.mean(np.abs(e[:,1]-e[:,0])/np.maximum(e[:,0],1e-3)):.2f}")

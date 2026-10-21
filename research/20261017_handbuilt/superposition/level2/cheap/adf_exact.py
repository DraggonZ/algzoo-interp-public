"""Same test as adf_tie.py, with each step's Gaussian projection done exactly (moments of relu(u) for Gaussian u by
Monte Carlo from 400k draws) instead of the gain approximation: separates 'covariance approximation' from 'Gaussian
state' as the cause of the failure."""
import sys, numpy as np, torch
sys.path.insert(0, "../../../family")
from common import ARC, trained, scores
from linelib import DEV, dt
Phi = torch.special.ndtr
def adf_exact(W, xm, xv, g):
    wi, wh, wo = W; m = torch.zeros(16, device=DEV, dtype=dt); C = torch.zeros(16, 16, device=DEV, dtype=dt)
    for t in range(10):
        um = wh @ m + xm[t] * wi; U = wh @ C @ wh.T + xv[t] * torch.outer(wi, wi)
        ev, V = torch.linalg.eigh(U); Lc = V * ev.clamp_min(0).sqrt()
        H = torch.relu(um + torch.randn(400000, 16, device=DEV, dtype=dt, generator=g) @ Lc.T)
        m = H.mean(0); C = torch.cov(H.T)
    return wo @ m, wo @ C @ wo.T
rng = np.random.default_rng(7)
for name, W in (("ARC", ARC), ("114", trained(114))):
    g = torch.Generator(device=DEV).manual_seed(0); res = []
    for trial in range(12):
        pL, pS, pT = rng.choice(10, 3, replace=False); L = float(rng.uniform(1.2, 2.2)); u = float(rng.uniform(0.4, 1.2))
        rest = [i for i in range(10) if i not in (pL, pS, pT)]
        a = Phi(torch.tensor(u, dtype=dt)).item(); zm = -np.exp(-u * u / 2) / np.sqrt(2 * np.pi) / a; zv = 1 - u * (-zm) - zm ** 2
        xm = torch.zeros(10, device=DEV, dtype=dt); xv = torch.zeros(10, device=DEV, dtype=dt)
        xm[rest] = zm; xv[rest] = zv; xm[pL] = L; xm[pS] = u; xm[pT] = u
        sm, sC = adf_exact(W, xm, xv, g); lm = (sm[pS] - sm[pT]).item(); lsd = (sC[pS, pS] + sC[pT, pT] - 2 * sC[pS, pT]).clamp_min(0).sqrt().item()
        n = 20000; Z = torch.special.ndtri(torch.rand(n, 7, device=DEV, dtype=dt, generator=g) * a)
        X = xm.repeat(n, 1); X[:, rest] = Z; s = scores(W, X); lam = s[:, pS] - s[:, pT]
        res.append((lam.mean().item(), lam.std().item(), lm, lsd))
    r = np.array(res)
    print(f"{name}: lean mean MC vs exact-projection Gaussian: " + ", ".join(f"{a:+.2f}/{c:+.2f}" for a, b, c, d in r))
    print(f"      lean sd    MC vs exact-projection Gaussian: " + ", ".join(f"{b:.2f}/{d:.2f}" for a, b, c, d in r))
    print(f"      median ratio of sd {np.median(r[:,3]/r[:,1]):.1f}; RMS error of mean {np.sqrt(np.mean((r[:,0]-r[:,2])**2)):.2f} (MC sd of lean, median {np.median(r[:,1]):.2f})")

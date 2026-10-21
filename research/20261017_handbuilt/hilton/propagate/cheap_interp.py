"""Cheaper sampling-free estimate, idea 1: run the network only on a coarse grid of (third u, gap a, gap b) per
position triple, record the margin (second-largest's score minus the best other), and fill in between grid points by
trilinear interpolation of the margin on a fine sub-grid (no network runs). Cost = network runs at the coarse nodes.
Gaps use a = R s^2 so nodes are dense near ties. Compared with the converged estimate (95.32%) on ARC's network."""
import itertools, time
import numpy as np, torch
from scipy.stats import norm
import adf
DEV = "cuda"; dt = torch.float64
T = torch.tensor(list(itertools.permutations(range(10), 3)), device=DEV)


def margins(W, pos, val):
    wi, wh, wo = W; n = pos.shape[0]; ar = torch.arange(n, device=DEV)
    bmu, _ = adf.background(val[:, 2]); X = bmu[:, None].repeat(1, 10)
    for k in range(3): X[ar, pos[:, k]] = val[:, k]
    h = torch.zeros(n, wi.shape[0], device=DEV, dtype=dt)
    for t in range(10): h = torch.relu(h @ wh.T + X[:, t:t + 1] * wi)
    S = h @ wo.T; s2 = S[ar, pos[:, 1]]; S[ar, pos[:, 1]] = -1e30
    return s2 - S.max(1).values


def estimate(W, nu, na, sub=6, ulo=-2.0, uhi=3.5, R=4.0):
    u = np.linspace(ulo, uhi, nu); s = np.linspace(0, 1, na); a = R * s ** 2
    U, A, B = np.meshgrid(u, a, a, indexing="ij")
    V = np.c_[(U + A + B).ravel(), (U + A).ravel(), U.ravel()]
    nV = len(V); Vt = torch.tensor(V, device=DEV, dtype=dt)
    pos = T.repeat_interleave(nV, 0); val = Vt.repeat(len(T), 1)
    Mg = margins(W, pos, val).reshape(len(T), nu, na, na)                   # [720, nu, na, na]
    # fine sub-grid inside each cell: trilinear interpolation of the margin, exact density weights
    f = (torch.arange(sub, device=DEV, dtype=dt) + 0.5) / sub
    tot = torch.zeros((), device=DEV, dtype=dt); mass = torch.zeros((), device=DEV, dtype=dt)
    uu = torch.tensor(u, device=DEV, dtype=dt); ss = torch.tensor(s, device=DEV, dtype=dt)
    for i in range(nu - 1):
        ui = uu[i] + f * (uu[i + 1] - uu[i]); du = (uu[i + 1] - uu[i]) / sub
        for j in range(na - 1):
            sj = ss[j] + f * (ss[j + 1] - ss[j]); aj = R * sj ** 2; daj = R * 2 * sj * (ss[j + 1] - ss[j]) / sub
            for k in range(na - 1):
                sk = ss[k] + f * (ss[k + 1] - ss[k]); bk = R * sk ** 2; dbk = R * 2 * sk * (ss[k + 1] - ss[k]) / sub
                c = Mg[:, i:i + 2, j:j + 2, k:k + 2]                         # [720,2,2,2]
                x, y, z = f[:, None, None], f[None, :, None], f[None, None, :]
                m = sum(c[:, p, q, r][:, None, None, None] * (x if p else 1 - x) * (y if q else 1 - y) * (z if r else 1 - z)
                        for p in (0, 1) for q in (0, 1) for r in (0, 1))
                m3 = ui[:, None, None]; m2 = m3 + aj[None, :, None]; m1 = m2 + bk[None, None, :]
                ncdf = lambda v: 0.5 * (1 + torch.erf(v / np.sqrt(2)))
                npdf = lambda v: torch.exp(-v * v / 2) / np.sqrt(2 * np.pi)
                w = npdf(m1) * npdf(m2) * npdf(m3) * ncdf(m3) ** 7 * du * daj[None, :, None] * dbk[None, None, :]
                tot += ((m > 0).to(dt) * w).sum(); mass += w.sum() * len(T)
    return (tot / mass).item(), mass.item(), len(T) * nV


if __name__ == "__main__":
    W = adf.net(None)
    for nu, na in [(4, 4), (5, 5), (6, 6), (8, 8), (10, 10), (14, 14), (20, 20)]:
        t0 = time.time(); e, mass, cost = estimate(W, nu, na)
        print(f"nodes {nu} x {na} x {na}: {cost:>9,d} runs  estimate {100*e:.3f}%  error {100*e-95.32:+.3f}  (mass {mass:.4f})  {time.time()-t0:.0f}s", flush=True)

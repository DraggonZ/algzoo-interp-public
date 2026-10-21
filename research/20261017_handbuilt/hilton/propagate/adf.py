"""Propagation of a probability model of ARC's network's hidden state, with no network runs (Hilton's suggestion).

Given the three largest numbers (values m1 > m2 > m3 and their positions), the other seven numbers are independent
normals truncated below m3. The hidden state is carried forward as a Gaussian (mean and full covariance) through the
real weights. Each ReLU is matched in mean and variance exactly; covariances between neurons are scaled by each
neuron's chance of being on. At the end the scores are Gaussian, and the chance that the second-largest's position wins
is a 9-dimensional normal orthant probability, integrated on fixed quasi-random points.
"""
import numpy as np
import torch

DEV = "cuda"
SQ2 = np.sqrt(2.0)


def ncdf(a): return 0.5 * (1 + torch.erf(a / SQ2))
def npdf(a): return torch.exp(-a * a / 2) / np.sqrt(2 * np.pi)


def background(m3):
    """Mean and variance of a standard normal truncated below m3."""
    r = npdf(m3) / ncdf(m3).clamp_min(1e-30)
    return -r, (1 - m3 * r - r * r).clamp_min(0)


def relu_moments(mu, cov, cross="gain"):
    """Mean and covariance of ReLU(a) for a ~ N(mu, cov): exact per-neuron mean and variance; off-diagonal covariance
    scaled by the two neurons' chances of being on (exact when one of them is always on)."""
    var = torch.diagonal(cov, dim1=-2, dim2=-1).clamp_min(0)
    sd = var.sqrt().clamp_min(1e-9)
    al = mu / sd
    P, f = ncdf(al), npdf(al)
    m = mu * P + sd * f
    e2 = (mu * mu + var) * P + mu * sd * f
    v = (e2 - m * m).clamp_min(0)
    if cross == "gain":
        C = cov * P[..., :, None] * P[..., None, :]
    else:
        C = torch.zeros_like(cov)
    C = C - torch.diag_embed(torch.diagonal(C, dim1=-2, dim2=-1)) + torch.diag_embed(v)
    return m, C


def propagate(W, pos, val, bg_var=True, cross="gain", wnoise=None):
    """pos [n,3] long, val [n,3] (m1, m2, m3). Returns the mean [n,10] and covariance [n,10,10] of the scores.
    wnoise: optional additive weight-noise SD (each weight + N(0, s^2)), entered as fresh independent noise per step."""
    wi, wh, wo = W
    n = pos.shape[0]; d = wi.shape[0]
    bmu, bvar = background(val[:, 2])
    if not bg_var:
        bvar = torch.zeros_like(bvar)
    mu = torch.zeros(n, d, device=DEV, dtype=val.dtype); cov = torch.zeros(n, d, d, device=DEV, dtype=val.dtype)
    for t in range(10):
        xm = bmu.clone(); xv = bvar.clone()
        for k in range(3):
            hit = pos[:, k] == t
            xm = torch.where(hit, val[:, k], xm); xv = torch.where(hit, torch.zeros_like(xv), xv)
        am = mu @ wh.T + xm[:, None] * wi
        ac = wh @ cov @ wh.T + xv[:, None, None] * (wi[:, None] * wi[None, :])
        if wnoise is not None:
            h2 = (mu * mu).sum(1) + torch.diagonal(cov, dim1=-2, dim2=-1).sum(1) + xm * xm + xv
            ac = ac + (wnoise ** 2 * h2)[:, None, None] * torch.eye(d, device=DEV, dtype=val.dtype)
        mu, cov = relu_moments(am, ac, cross)
    sm = mu @ wo.T; sc = wo @ cov @ wo.T
    if wnoise is not None:
        h2 = (mu * mu).sum(1) + torch.diagonal(cov, dim1=-2, dim2=-1).sum(1)
        sc = sc + (wnoise ** 2 * h2)[:, None, None] * torch.eye(10, device=DEV, dtype=val.dtype)
    return sm, sc


_Z = {}
def orthant(sm, sc, p2, nq=4096):
    """Chance that score p2 beats the other nine, for Gaussian scores (fixed Sobol normal points)."""
    n = sm.shape[0]
    if nq not in _Z:
        e = torch.quasirandom.SobolEngine(9, scramble=True, seed=7).draw(nq).to(DEV, torch.float64)
        _Z[nq] = (SQ2 * torch.erfinv(2 * e.clamp(1e-9, 1 - 1e-9) - 1))
    Z = _Z[nq].to(sm.dtype)
    others = torch.stack([torch.tensor([k for k in range(10) if k != j]) for j in range(10)]).to(DEV)[p2]  # [n,9]
    A = torch.zeros(n, 9, 10, device=DEV, dtype=sm.dtype)
    A[torch.arange(n)[:, None], torch.arange(9)[None], p2[:, None].expand(n, 9)] = 1.0
    A.scatter_(2, others[:, :, None], -1.0)
    lm = (A @ sm[:, :, None])[..., 0]; lc = A @ sc @ A.transpose(1, 2)
    Lc = torch.linalg.cholesky(lc + 1e-6 * torch.eye(9, device=DEV, dtype=sm.dtype) * (1 + torch.diagonal(lc, dim1=-2, dim2=-1).mean(1))[:, None, None])
    leads = lm[:, None, :] + Z[None] @ Lc.transpose(1, 2)                     # [n, nq, 9]
    return (leads > 0).all(2).to(sm.dtype).mean(1), lm, lc


def net(W):
    from algzoo.core import example
    return tuple(torch.tensor(np.asarray(p, float), device=DEV, dtype=torch.float64) for p in (W if W is not None else example()))

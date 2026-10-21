"""Per-condition models of P(right | three largest) from weights only (no sampled lists).

All models run the network on fixed inputs and use its exact local linear map (Jacobian) with respect to the smaller
numbers that are still treated as random; their effect on the chosen leads is taken as Gaussian.
  two     : seven smaller numbers at their mean (normal truncated below the third); leads over the largest and over the
            third, Gaussian with covariance sum_s var_s g_s g_s^T.  Losses to smaller numbers ignored.
  two+1d  : 'two' times, for each smaller position k, P(lead over k > 0) with x_k on quadrature nodes (exact runs) and
            the other six as Gaussian through the Jacobian.
  fourth  : the fourth largest exact (its position: each of seven; its value: Q4 nodes of its exact conditional law),
            the other six at their mean truncated below it, Gaussian through the Jacobian; leads over the largest, the
            third and the fourth.  Losses to the fifth and below ignored.
Each returns P(right) and the split of wrong answers [largest, third, smaller] (from the Gaussian samples)."""
import numpy as np, torch
from lib import *

_ZZ = {}
def gauss_split(lm, lc, nq=512):
    """lm [n,c], lc [n,c,c]; returns P(all > 0) and P(lead j is the most negative and < 0) for each j."""
    c = lm.shape[1]
    if c not in _ZZ:
        e = torch.quasirandom.SobolEngine(c, scramble=True, seed=11).draw(nq).to(DEV, dt)
        _ZZ[c] = SQ2 * torch.erfinv(2 * e.clamp(1e-9, 1 - 1e-9) - 1)
    I = torch.eye(c, device=DEV, dtype=dt)
    Lc = torch.linalg.cholesky(lc + I * (1e-10 + 1e-8 * torch.diagonal(lc, dim1=-2, dim2=-1).mean(1))[:, None, None])
    L = lm[:, None, :].float() + _ZZ[c][None].float() @ Lc.transpose(1, 2).float()   # [n,nq,c], float32 for speed
    good = (L > 0).all(2); ok = good.to(dt).mean(1)
    j = L.argmin(2); bad = ~good
    split = torch.stack([((j == q) & bad).to(dt).mean(1) for q in range(c)], 1)
    return ok, split


def lead_moments(W, X, var, comps, p2):
    """X [n,10] inputs, var [n,10] variance of each input treated as random (0 = fixed), comps [n,c] competitor
    positions. Leads score[p2]-score[comp]: value at X, covariance through the exact local linear map."""
    S, G = run_jac(W, X); n = X.shape[0]; ar = torch.arange(n, device=DEV)
    lm = S[ar, p2][:, None] - torch.gather(S, 1, comps)
    g = G[ar, :, p2][:, :, None] - torch.gather(G, 2, comps[:, None, :].expand(-1, 10, -1))   # [n,10 inputs,c]
    lc = (g * var[:, :, None]).transpose(1, 2) @ g
    return lm, lc


def tvar(m):
    """Variance of N(0,1) truncated below m."""
    r = -background_mean(m); return (1 - m * r - r * r).clamp_min(0)


def model_two(W, pos, val, with1d=False, Q=16):
    X0, isbg = base_inputs(pos, val); n = len(pos)
    var = isbg.to(dt) * tvar(val[:, 2])[:, None]
    lm, lc = lead_moments(W, X0, var, pos[:, [0, 2]], pos[:, 1])
    ok, split = gauss_split(lm, lc)
    small = torch.zeros(n, device=DEV, dtype=dt)
    if with1d:
        nodes, w = bg_nodes(val[:, 2], Q); pk = torch.ones(n, device=DEV, dtype=dt)
        for k in range(10):
            Xk = X0[:, None, :].repeat(1, Q, 1); Xk[:, :, k] = nodes; Xk = Xk.reshape(-1, 10)
            vk = var.clone(); vk[:, k] = 0; vk = vk.repeat_interleave(Q, 0)
            comp = torch.full((n * Q, 1), k, device=DEV)
            lmk, lck = lead_moments(W, Xk, vk, comp, pos[:, 1].repeat_interleave(Q))
            p = ncdf(lmk[:, 0] / lck[:, 0, 0].clamp_min(1e-30).sqrt()).reshape(n, Q) @ w
            pk = pk * torch.where(isbg[:, k], p, torch.ones_like(p))
        small = ok * (1 - pk); ok = ok * pk
    return ok, torch.cat([split, small[:, None]], 1)


def model_fourth(W, pos, val, Q4=8, inc_small=False):
    """inc_small: Gaussian over all nine leads (the other six smaller numbers' leads included)."""
    """The fourth largest exact over quadrature; returns P(right) and split [largest, third, smaller]."""
    n = len(pos); X0, isbg = base_inputs(pos, val)
    x, w = np.polynomial.legendre.leggauss(Q4); q = torch.tensor(0.5 * (x + 1), device=DEV, dtype=dt); w = torch.tensor(0.5 * w, device=DEV, dtype=dt)
    x4 = nicdf((ncdf(val[:, 2])[:, None] * q[None] ** (1 / 7)).clamp(1e-300, 1 - 1e-16))   # [n,Q4]
    bgpos = torch.nonzero(isbg)[:, 1].reshape(n, 7)                                           # [n,7]
    # inputs [n,7,Q4,10]
    X = X0[:, None, None, :].repeat(1, 7, Q4, 1)
    m6 = background_mean(x4)                                                                   # [n,Q4]
    bgmask = isbg[:, None, None, :].expand(n, 7, Q4, 10)
    X = torch.where(bgmask, m6[:, None, :, None].expand(n, 7, Q4, 10), X)
    X.scatter_(3, bgpos[:, :, None, None].expand(n, 7, Q4, 1), x4[:, None, :, None].expand(n, 7, Q4, 1))
    var = bgmask.to(dt) * tvar(x4)[:, None, :, None]
    var = var.scatter(3, bgpos[:, :, None, None].expand(n, 7, Q4, 1), 0.0)
    comps = torch.stack([pos[:, 0, None].expand(n, 7), pos[:, 2, None].expand(n, 7), bgpos], 2)
    if inc_small:   # append the other six smaller positions
        rest = torch.stack([torch.cat([bgpos[:, :j], bgpos[:, j + 1:]], 1) for j in range(7)], 1)   # [n,7,6]
        comps = torch.cat([comps, rest], 2)
    c = comps.shape[2]; comps = comps[:, :, None, :].expand(n, 7, Q4, c)
    p2 = pos[:, 1, None, None].expand(n, 7, Q4)
    lm, lc = lead_moments(W, X.reshape(-1, 10), var.reshape(-1, 10), comps.reshape(-1, c), p2.reshape(-1))
    ok, split = gauss_split(lm, lc)
    split = torch.cat([split[:, :2], split[:, 2:].sum(1, keepdim=True)], 1)
    wt = (w / 7)[None, None, :].expand(n, 7, Q4).reshape(-1)
    ok = (ok * wt).reshape(n, -1).sum(1); split = (split * wt[:, None]).reshape(n, -1, 3).sum(1)
    return ok, split


def _lead_vals(W, X, comps, p2):
    S = run(W, X); ar = torch.arange(X.shape[0], device=DEV)
    return S[ar, p2][:, None] - torch.gather(S, 1, comps)


def model_fourth_m(W, pos, val, Q4=4, moments="axis", Qa=3, M=16, inc_small=False, small1d=False):
    """Like model_fourth (three leads), but the six smaller numbers' effect on the leads from exact network runs instead
    of the Jacobian at their mean:
      axis : each of the six varied alone over Qa nodes of its truncated law (others at the mean); mean = f0 + sum of
             per-axis shifts, covariance = sum of per-axis covariances (additive, switches along each axis exact)
      qmc  : M fixed scrambled-Sobol points of the six jointly (a deterministic cubature); mean and covariance of the
             leads over them."""
    n = len(pos); X0, isbg = base_inputs(pos, val)
    x, w = np.polynomial.legendre.leggauss(Q4); q = torch.tensor(0.5 * (x + 1), device=DEV, dtype=dt); w = torch.tensor(0.5 * w, device=DEV, dtype=dt)
    x4 = nicdf((ncdf(val[:, 2])[:, None] * q[None] ** (1 / 7)).clamp(1e-300, 1 - 1e-16))
    bgpos = torch.nonzero(isbg)[:, 1].reshape(n, 7)
    X = X0[:, None, None, :].repeat(1, 7, Q4, 1); m6 = background_mean(x4)
    bgmask = isbg[:, None, None, :].expand(n, 7, Q4, 10)
    X = torch.where(bgmask, m6[:, None, :, None].expand(n, 7, Q4, 10), X)
    X.scatter_(3, bgpos[:, :, None, None].expand(n, 7, Q4, 1), x4[:, None, :, None].expand(n, 7, Q4, 1))
    rnd = bgmask.clone(); rnd.scatter_(3, bgpos[:, :, None, None].expand(n, 7, Q4, 1), False)       # the six random slots
    comps = torch.stack([pos[:, 0, None].expand(n, 7), pos[:, 2, None].expand(n, 7), bgpos], 2)
    if inc_small or small1d:
        rest = torch.stack([torch.cat([bgpos[:, :j], bgpos[:, j + 1:]], 1) for j in range(7)], 1); comps = torch.cat([comps, rest], 2)
    c = comps.shape[2]; comps = comps[:, :, None, :].expand(n, 7, Q4, c).reshape(-1, c)
    p2 = pos[:, 1, None, None].expand(n, 7, Q4).reshape(-1)
    X = X.reshape(-1, 10); rnd = rnd.reshape(-1, 10); x4f = x4[:, None, :].expand(n, 7, Q4).reshape(-1); N = X.shape[0]
    slots = torch.nonzero(rnd)[:, 1].reshape(N, 6)
    if moments == "axis":
        nodes, wq = bg_nodes(x4f, Qa)                                                     # [N,Qa]
        L0 = _lead_vals(W, X, comps, p2); mean = L0.clone(); cov = torch.zeros(N, c, c, device=DEV, dtype=dt); own, sh, vv = [], [], []
        for j in range(6):
            Xj = X[:, None, :].repeat(1, Qa, 1); Xj.scatter_(2, slots[:, j, None, None].expand(N, Qa, 1), nodes[:, :, None])
            Lj = _lead_vals(W, Xj.reshape(-1, 10), comps.repeat_interleave(Qa, 0), p2.repeat_interleave(Qa)).reshape(N, Qa, c)
            mj = (wq[None, :, None] * Lj).sum(1); d = Lj - mj[:, None]
            mean += mj - L0; cov += (wq[None, :, None, None] * d[..., :, None] * d[..., None, :]).sum(1)
            if small1d: own.append(Lj[:, :, 3 + j]); sh.append(mj[:, 3 + j] - L0[:, 3 + j]); vv.append(torch.diagonal(cov, dim1=1, dim2=2)[:, 3 + j].clone())
    else:
        e = torch.quasirandom.SobolEngine(6, scramble=True, seed=3).draw(M).to(DEV, dt)      # [M,6] fixed
        U = (e[None] * ncdf(x4f)[:, None, None]).clamp_min(1e-300); Z = nicdf(U)          # [N,M,6]
        Xm = X[:, None, :].repeat(1, M, 1); Xm.scatter_(2, slots[:, None, :].expand(N, M, 6), Z)
        Lm = _lead_vals(W, Xm.reshape(-1, 10), comps.repeat_interleave(M, 0), p2.repeat_interleave(M)).reshape(N, M, c)
        mean = Lm.mean(1); d = Lm - mean[:, None]; cov = (d[..., :, None] * d[..., None, :]).mean(1)
    if small1d:
        # the three main leads jointly Gaussian; each of the six other smaller numbers' lead: exact along its own axis,
        # plus the other axes' mean shift and variance as Gaussian; treated as independent of each other.
        ok, split = gauss_split(mean[:, :3], cov[:, :3, :3])
        vd = torch.diagonal(cov, dim1=1, dim2=2)[:, 3:]
        Pk = torch.ones(N, device=DEV, dtype=dt)
        for k in range(6):
            tot_shift = (mean[:, 3 + k] - L0[:, 3 + k]) - sh[k]; own_var = ((wq[None] * (own[k] - (wq[None] * own[k]).sum(1, keepdim=True)) ** 2).sum(1))
            rest_var = (vd[:, k] - own_var).clamp_min(1e-12)
            Pk = Pk * (wq[None] * ncdf((own[k] + tot_shift[:, None]) / rest_var.sqrt()[:, None])).sum(1)
        split = torch.cat([split[:, :2], (ok * (1 - Pk))[:, None]], 1); ok = ok * Pk
    else:
        ok, split = gauss_split(mean, cov); split = torch.cat([split[:, :2], split[:, 2:].sum(1, keepdim=True)], 1)
    wt = (w / 7)[None, None, :].expand(n, 7, Q4).reshape(-1)
    return (ok * wt).reshape(n, -1).sum(1), (split * wt[:, None]).reshape(n, -1, 3).sum(1)

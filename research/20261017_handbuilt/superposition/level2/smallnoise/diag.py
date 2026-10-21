"""Diagnosis per condition (fixed three largest): which Gaussian model of the seven smaller numbers' effect on the nine
leads gives the right chance of a correct answer? Truth by sampling 4000 backgrounds per condition (diagnosis only).

Conditions: 300 from random lists, 300 from random lists with a near tie (second-third or first-second gap < 0.1).
Models of the leads, all Gaussian, integrated as a 9-d normal orthant:
  mean0/1  : the old grid estimate (seven at their mean, 0/1 answer)
  jacobian : mean at the mean path, covariance var_b * G G^T, G the exact local linear map there
  additive : each of the seven varied alone over 16 quadrature nodes (others at the mean), exact network runs;
             mean = f0 + sum_s (E f_s - f0), covariance = sum_s Cov f_s   (first-order ANOVA, pattern switches per input)
  oracle   : mean and covariance of the leads from the 4000 sampled backgrounds (best any Gaussian model can do)
"""
import sys, numpy as np, torch
from lib import *
W = load_net(sys.argv[1] if len(sys.argv) > 1 else "ARC")
rng = np.random.default_rng(3); K = 4000


def conditions(n, near):
    out = []
    while len(out) < n:
        X0 = rng.standard_normal((20000, 10)); o = np.argsort(-X0, 1)[:, :3]; v = np.take_along_axis(X0, o, 1)
        if near: m = ((v[:, 1] - v[:, 2]) < 0.1) | ((v[:, 0] - v[:, 1]) < 0.1)
        else: m = np.ones(len(v), bool)
        out += list(zip(o[m], v[m]))
    o, v = zip(*out[:n]); return np.array(o), np.array(v)


def truth_and_oracle(pos, val):
    n = len(pos); tr = np.zeros(n); lm = torch.zeros(n, 9, device=DEV, dtype=dt); lc = torch.zeros(n, 9, 9, device=DEV, dtype=dt)
    A, _ = lead_matrix(pos[:, 1])
    for i in range(n):
        p = ncdf(val[i, 2]).item()
        U = torch.rand(K, 10, device=DEV, dtype=dt) * p; X = nicdf(U.clamp_min(1e-300))
        X[:, pos[i]] = val[i]
        L = run(W, X) @ A[i].T
        tr[i] = (L > 0).all(1).double().mean().item(); lm[i] = L.mean(0); lc[i] = torch.cov(L.T)
    return tr, lm, lc


def additive(pos, val, Q=16):
    n = pos.shape[0]; X0, isbg = base_inputs(pos, val); A, _ = lead_matrix(pos[:, 1])
    L0 = (A @ run(W, X0)[:, :, None])[..., 0]
    nodes, w = bg_nodes(val[:, 2], Q)
    mean = L0.clone(); cov = torch.zeros(n, 9, 9, device=DEV, dtype=dt)
    for s in range(10):
        Xs = X0[:, None, :].repeat(1, Q, 1); Xs[:, :, s] = nodes
        Ls = (A[:, None] @ run(W, Xs.reshape(-1, 10)).reshape(n, Q, 10, 1))[..., 0]   # [n,Q,9]
        m = (w[None, :, None] * Ls).sum(1); dL = Ls - m[:, None]
        c = (w[None, :, None, None] * dL[..., :, None] * dL[..., None, :]).sum(1)
        b = isbg[:, s].to(dt)
        mean += b[:, None] * (m - L0); cov += b[:, None, None] * c
    return L0, mean, cov


def jacobian(pos, val):
    X0, isbg = base_inputs(pos, val); A, _ = lead_matrix(pos[:, 1])
    S, G = run_jac(W, X0); G = G * isbg[:, :, None].to(dt)
    nodes, w = bg_nodes(val[:, 2], 32); m = (nodes * w).sum(1); var = (nodes ** 2 * w).sum(1) - m ** 2
    GL = G @ A.transpose(1, 2)                                     # [n,10 inputs,9 leads]
    return (A @ S[:, :, None])[..., 0], var[:, None, None] * GL.transpose(1, 2) @ GL


if __name__ == "__main__":
    torch.manual_seed(0)
    for label, near in [("random conditions", False), ("near-tie conditions", True)]:
        o, v = conditions(300, near); pos = torch.tensor(o, device=DEV); val = torch.tensor(v, device=DEV, dtype=dt)
        tr, olm, olc = truth_and_oracle(pos, val)
        L0, am, ac = additive(pos, val); jm, jc = jacobian(pos, val)
        res = {"mean0/1": (L0 > 0).all(1).double().cpu().numpy(),
               "jacobian": orthant(jm, jc).cpu().numpy(),
               "additive": orthant(am, ac).cpu().numpy(),
               "additive, mean only 0/1": (am > 0).all(1).double().cpu().numpy(),
               "oracle Gaussian": orthant(olm, olc).cpu().numpy()}
        print(f"{label}: truth mean P(right) {tr.mean():.4f}")
        for k, r in res.items():
            print(f"  {k:26s} mean {r.mean():.4f}  bias {r.mean()-tr.mean():+.4f}  mean abs error {np.abs(r-tr).mean():.4f}  corr {np.corrcoef(r, tr)[0,1]:.3f}")
        # how well the additive model gets the moments
        sd_o = torch.diagonal(olc, dim1=-2, dim2=-1).sqrt(); sd_a = torch.diagonal(ac, dim1=-2, dim2=-1).sqrt()
        j = olm.argmin(1, keepdim=True)  # the closest competitor (by oracle mean)
        g = lambda t: t.gather(1, j)[:, 0].cpu().numpy()
        print(f"  closest competitor: oracle mean vs additive mean: median |diff| {np.median(np.abs(g(olm)-g(am))):.3f} (oracle sd median {np.median(g(sd_o)):.3f}); "
              f"sd ratio additive/oracle median {np.median(g(sd_a)/g(sd_o)):.3f}, 10-90% {np.percentile(g(sd_a)/g(sd_o),10):.2f}-{np.percentile(g(sd_a)/g(sd_o),90):.2f}")
        bad = np.argsort(-np.abs(res["additive"] - tr))[:5]
        for i in bad:
            print(f"    worst additive: positions {o[i]} values {v[i].round(2)} truth {tr[i]:.3f} additive {res['additive'][i]:.3f} oracle {res['oracle Gaussian'][i]:.3f} "
                  f"lead(closest) oracle {g(olm)[i]:.2f}+-{g(sd_o)[i]:.2f} additive {g(am)[i]:.2f}+-{g(sd_a)[i]:.2f}")

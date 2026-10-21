"""Calibration pilot: per-position-cell mean input + delta-method width (no sampling of the model).

For each cell (p1, p2) [or (p1, p2, p3)], set the top values to their order-statistic means and every
lower value to the common mean of the lower order statistics. One forward and one backward pass give
the margin mu = logit[p2] - max other logit and its gradient g. Width s^2 = g' Sigma g, with Sigma the
exact covariance of x given the cell (from order-statistic moments, model-independent constants).
Estimate = mean over cells of Phi(mu / s); variant '0/1' uses 1[mu > 0].
"""
import sys

sys.path.insert(0, "/data/output/research/algzoo/20261016_m16_10/scripts")
import itertools

import numpy as np
import torch as th
from scipy.stats import norm

from bench import load_bench, FWD_OPS
from core import example

n = 10
# model-independent constants: order-statistic moments of 10 iid N(0,1) (largest first)
rng = np.random.default_rng(0)
S = -np.sort(-rng.standard_normal((10**7, n)), axis=1)
MU = S.mean(0)
M2 = (S.T @ S) / S.shape[0]
del S


def cell_moments(k):
    """Mean and covariance of x given the positions of the top k values (positions 0..k-1 hold them here)."""
    low = list(range(k, n))
    mean = np.concatenate([MU[:k], np.full(n - k, MU[low].mean())])
    E = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i < k and j < k:
                E[i, j] = M2[i, j]
            elif i < k:
                E[i, j] = M2[i, low].mean()
            elif j < k:
                E[i, j] = M2[low, j].mean()
            elif i == j:
                E[i, j] = np.mean([M2[a, a] for a in low])
            else:
                E[i, j] = np.mean([M2[a, b] for a in low for b in low if a != b])
    return mean, E - np.outer(mean, mean)


def estimate(params, k=2):
    w_hi, w_hh, w_oh = (th.tensor(p) for p in params)
    mean_c, cov_c = cell_moments(k)
    cells = [c for c in itertools.permutations(range(n), k)]
    X = th.zeros(len(cells), n, dtype=th.float64)
    perm = np.zeros((len(cells), n), dtype=int)   # perm[c, pos] = index into cell-ordered vector
    for ci, c in enumerate(cells):
        rest = [p for p in range(n) if p not in c]
        order = list(c) + rest
        for slot, pos in enumerate(order):
            perm[ci, pos] = slot
        X[ci] = th.tensor(mean_c[perm[ci]])
    X.requires_grad_(True)
    h = th.zeros(len(cells), w_hh.shape[0], dtype=th.float64)
    for t in range(n):
        h = th.relu(h @ w_hh.T + X[:, t, None] * w_hi)
    logits = h @ w_oh.T
    tgt = th.tensor([c[1] for c in cells])
    lt = logits.gather(1, tgt[:, None])[:, 0]
    other = logits.scatter(1, tgt[:, None], -th.inf).max(1).values
    mu = lt - other
    g, = th.autograd.grad(mu.sum(), X)
    mu = mu.detach().numpy(); g = g.numpy()
    s = np.array([np.sqrt(g[ci] @ cov_c[np.ix_(perm[ci], perm[ci])] @ g[ci]) for ci in range(len(cells))])
    ops = len(cells) * 3 * FWD_OPS
    return norm.cdf(mu / np.maximum(s, 1e-12)).mean(), (mu > 0).mean(), ops


if __name__ == "__main__":
    names, acc, models = load_bench(("mult",))
    for k in (2, 3):
        res = np.array([estimate(m, k)[:2] for m in models])
        ops = estimate(models[0], k)[2]
        neq = ops / FWD_OPS
        samp = np.sqrt(np.mean(acc * (1 - acc)) / neq)
        b = estimate(example(), k)
        print(f"cells by top-{k} positions ({len(list(itertools.permutations(range(n), k)))} cells), cost ~{neq:.0f} samples, sampling RMSE {samp:.4f}")
        print(f"  base model: probit {b[0]:.3f}, 0/1 {b[1]:.3f} (true 0.953)")
        for j, lab in enumerate(["probit", "0/1"]):
            e = res[:, j]
            print(f"  {lab:6s}: RMSE {np.sqrt(np.mean((e-acc)**2)):.4f}, bias {np.mean(e-acc):+.4f}, corr {np.corrcoef(e,acc)[0,1]:.3f}")
            for g in ("mult0.005", "mult0.01", "mult0.02"):
                s_ = np.array([str(x).startswith(g + "_") for x in names])
                print(f"     {g}: RMSE {np.sqrt(np.mean((e[s_]-acc[s_])**2)):.4f} bias {np.mean(e[s_]-acc[s_]):+.4f} corr {np.corrcoef(e[s_],acc[s_])[0,1]:.2f}")

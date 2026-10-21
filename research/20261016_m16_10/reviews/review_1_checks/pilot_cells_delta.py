"""Diagnostic: is the per-cell *change* between base and noisy model right?
est_noisy = mean_c [ acc_base(c) + probit_noisy(c) - probit_base(c) ], clipped to [0,1] per cell.
acc_base(c) from sampling the base model only (advice), never the noisy copies."""
import sys
sys.path.insert(0, "/data/output/research/algzoo/20261016_m16_10/scripts")
import itertools
import numpy as np
import torch as th
from scipy.stats import norm
import pilot_cells as P
from core import example, forward, target

def per_cell_probit(params, k):
    w_hi, w_hh, w_oh = (th.tensor(p) for p in params)
    mean_c, cov_c = P.cell_moments(k)
    cells = list(itertools.permutations(range(10), k))
    X = th.zeros(len(cells), 10, dtype=th.float64); perm = np.zeros((len(cells), 10), dtype=int)
    for ci, c in enumerate(cells):
        order = list(c) + [p for p in range(10) if p not in c]
        for slot, pos in enumerate(order): perm[ci, pos] = slot
        X[ci] = th.tensor(mean_c[perm[ci]])
    X.requires_grad_(True)
    h = th.zeros(len(cells), 16, dtype=th.float64)
    for t in range(10): h = th.relu(h @ w_hh.T + X[:, t, None] * w_hi)
    lg = h @ w_oh.T; tgt = th.tensor([c[1] for c in cells])
    mu = lg.gather(1, tgt[:, None])[:, 0] - lg.scatter(1, tgt[:, None], -th.inf).max(1).values
    g, = th.autograd.grad(mu.sum(), X); mu = mu.detach().numpy(); g = g.numpy()
    s = np.array([np.sqrt(g[i] @ cov_c[np.ix_(perm[i], perm[i])] @ g[i]) for i in range(len(cells))])
    return norm.cdf(mu / np.maximum(s, 1e-12)), cells

k = 3
pb, cells = per_cell_probit(example(), k)
# base-model per-cell accuracy (advice), by sampling the base model
dev = "cuda"
with th.no_grad():
    x = th.randn(2**24, 10, device=dev, generator=th.Generator(dev).manual_seed(11))
    c = (forward(example(), x).argmax(-1) == target(x)).double()
    o = x.argsort(-1, descending=True)
    key = (o[:, 0] * 100 + o[:, 1] * 10 + o[:, 2])
    acc_cell = (th.bincount(key, weights=c, minlength=1000) / th.bincount(key, minlength=1000).clamp(min=1)).cpu().numpy()
ab = np.array([acc_cell[a*100+b*10+cc] for a, b, cc in cells])
print(f"base: per-cell corr(probit, sampled acc) {np.corrcoef(pb, ab)[0,1]:.3f}; mean probit {pb.mean():.4f} vs {ab.mean():.4f}")
names, acc, models = P.load_bench(("mult",))
est = np.array([np.clip(ab + per_cell_probit(m, k)[0] - pb, 0, 1).mean() for m in models])
print(f"advice-corrected: RMSE {np.sqrt(np.mean((est-acc)**2)):.4f} bias {np.mean(est-acc):+.4f} corr {np.corrcoef(est,acc)[0,1]:.3f}")
for gname in ("mult0.005", "mult0.01", "mult0.02"):
    s_ = np.array([str(v).startswith(gname + "_") for v in names])
    print(f"   {gname}: RMSE {np.sqrt(np.mean((est[s_]-acc[s_])**2)):.4f} bias {np.mean(est[s_]-acc[s_]):+.4f}")

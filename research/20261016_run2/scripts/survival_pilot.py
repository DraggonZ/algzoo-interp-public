"""Diagnostic: 'survive every low input separately'. With the top two at their values and one low input added back
(others zero), is the network still right? Surrogate: right iff right with the top two alone and with each of the
8 single additions. Also the per-input product of single-input survival rates per (pair) (independence).
Sampling-based diagnostic, not an estimator.
"""
import numpy as np
import torch as th

from algzoo.bench import load_bench
from algzoo.core import example, forward

dev = "cuda"
N = 2**19


@th.no_grad()
def run(params, seed):
    g = th.Generator(device=dev).manual_seed(seed)
    x = th.randn(N, 10, generator=g, device=dev)
    v, idx = x.sort(-1, descending=True)
    p2 = idx[:, 1]
    top2 = th.zeros_like(x).scatter_(1, idx[:, :2], v[:, :2])
    c_full = forward(params, x).argmax(-1) == p2
    c0 = forward(params, top2).argmax(-1) == p2
    xs = top2.unsqueeze(0).repeat(8, 1, 1)
    for k in range(8):
        xs[k].scatter_(1, idx[:, 2 + k:3 + k], v[:, 2 + k:3 + k])
    ck = forward(params, xs).argmax(-1) == p2            # [8, N]
    surv_all = c0 & ck.all(0)
    # independence version: P(right) ≈ P(c0) * Π_k P(ck | c0) per (p1, p2) cell
    key = idx[:, 0] * 10 + idx[:, 1]
    est = 0.0
    for cell in key.unique():
        m = key == cell
        pc0 = c0[m].float().mean()
        if pc0 == 0:
            continue
        cond = (ck[:, m] & c0[m]).float().sum(1) / c0[m].float().sum()
        est += m.float().mean() * pc0 * cond.prod()
    return c_full.float().mean().item(), surv_all.float().mean().item(), (surv_all == c_full).float().mean().item(), est.item()


names, acc, models = load_bench(("mult",))
b = run(example(), 0)
print(f"base: true {b[0]:.4f}, survive-all {b[1]:.4f} (agreement {b[2]:.3f}), per-pair independence {b[3]:.4f}")
R = np.array([run(m, 1 + i) for i, m in enumerate(models)])
sig = np.array([float(str(n).split("_")[0][4:]) for n in names])
for j, lab in [(1, "survive-all"), (3, "per-pair independence")]:
    e = R[:, j]
    print(f"{lab}: RMSE {np.sqrt(np.mean((e - acc)**2)):.4f} bias {np.mean(e - acc):+.4f} corr {np.corrcoef(e, acc)[0,1]:.3f};"
          + " ".join(f" σ={s}: RMSE {np.sqrt(np.mean((e[sig==s]-acc[sig==s])**2)):.4f} bias {np.mean(e[sig==s]-acc[sig==s]):+.4f}" for s in (0.005, 0.01, 0.02)))
print("survive-all per-input agreement with the network: mean", R[:, 2].mean().round(3))

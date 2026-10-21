"""Diagnostic variants of the survive-every-low-input surrogate (sampling-based, not an estimator).
V_all: right with top two and with each single low input added (others zero).
V_after: only low inputs after both spikes; V_between: only those after the first spike.
V_rep_k: each low input replaced by a representative value r_k(m2) (a fixed conditional quantile of x | x < m2), tested one at a time.
"""
import numpy as np
import torch as th

from algzoo.bench import load_bench
from algzoo.core import example, forward

dev = "cuda"
N = 2**19
QS = (0.5, 0.9, 0.97)


@th.no_grad()
def run(params, seed):
    g = th.Generator(device=dev).manual_seed(seed)
    x = th.randn(N, 10, generator=g, device=dev)
    v, idx = x.sort(-1, descending=True)
    p1, p2 = idx[:, 0], idx[:, 1]
    top2 = th.zeros_like(x).scatter_(1, idx[:, :2], v[:, :2])
    c_full = forward(params, x).argmax(-1) == p2
    c0 = forward(params, top2).argmax(-1) == p2
    qpos = idx[:, 2:]                                          # [N, 8] positions of the low inputs
    xs = top2.unsqueeze(0).repeat(8, 1, 1)
    for k in range(8):
        xs[k].scatter_(1, qpos[:, k:k + 1], v[:, 2 + k:3 + k])
    ck = forward(params, xs).argmax(-1) == p2                  # [8, N]
    after = qpos.T > th.maximum(p1, p2)
    between = qpos.T > th.minimum(p1, p2)
    out = {"V_all": (c0 & ck.all(0)), "V_after": (c0 & (ck | ~after).all(0)), "V_between": (c0 & (ck | ~between).all(0))}
    m2 = v[:, 1]
    nd = th.distributions.Normal(0.0, 1.0)
    for q in QS:
        rep = nd.icdf((q * th.special.ndtr(m2)).clamp(1e-7, 1 - 1e-7))   # q-quantile of x | x < m2
        xr = top2.unsqueeze(0).repeat(8, 1, 1)
        for k in range(8):
            xr[k].scatter_(1, qpos[:, k:k + 1], rep[:, None])
        cr = forward(params, xr).argmax(-1) == p2
        out[f"V_rep{q}"] = c0 & cr.all(0)
    return c_full.float().mean().item(), {k: val.float().mean().item() for k, val in out.items()}


names, acc, models = load_bench(("mult",))
res = [run(m, 1 + i) for i, m in enumerate(models)]
sig = np.array([float(str(n).split("_")[0][4:]) for n in names])
for key in res[0][1]:
    e = np.array([r[1][key] for r in res])
    print(f"{key:12s} RMSE {np.sqrt(np.mean((e - acc)**2)):.4f} bias {np.mean(e - acc):+.4f} corr {np.corrcoef(e, acc)[0,1]:.3f};"
          + " ".join(f" σ={s}: {np.sqrt(np.mean((e[sig==s]-acc[sig==s])**2)):.4f}" for s in (0.005, 0.01, 0.02)))
b = run(example(), 0)
print("base:", round(b[0], 4), {k: round(val, 4) for k, val in b[1].items()})

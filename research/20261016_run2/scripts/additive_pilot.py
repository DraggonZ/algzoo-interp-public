"""Additive-background pilot (sampling-based diagnostic, not an estimator).

Surrogate logits: L_add(x) = L(top2) + sum over low positions p of [L(top2 + x_p) - L(top2)],
where L(top2) runs the network with every input except the top two set to zero, and L(top2 + x_p)
adds back one low input. Also reported, as references: Z_2 (top two only), Z_3 (top three only), and a
mean-background surrogate (low inputs replaced by E[x | x < m2] = -φ(m2)/Φ(m2)).
Per copy: surrogate accuracy against the true accuracy (2^29 samples); per input: agreement with the network.
"""
import sys

import numpy as np
import torch as th

from algzoo.bench import load_bench
from algzoo.core import example, forward

dev = "cuda"
N = 2**19 if len(sys.argv) < 2 else int(sys.argv[1])


@th.no_grad()
def run(params, seed=0):
    g = th.Generator(device=dev).manual_seed(seed)
    x = th.randn(N, 10, generator=g, device=dev)
    v, idx = x.sort(-1, descending=True)
    p2 = idx[:, 1]
    top2 = th.zeros_like(x).scatter_(1, idx[:, :2], v[:, :2])
    L_full = forward(params, x)
    L0 = forward(params, top2)
    # single low input added back, all 8 at once as a batch [8, N, 10]
    low_idx = idx[:, 2:]                                   # [N, 8]
    xs = top2.unsqueeze(0).repeat(8, 1, 1)
    for k in range(8):
        xs[k].scatter_(1, low_idx[:, k:k + 1], v[:, 2 + k:3 + k])
    Lk = forward(params, xs)                               # [8, N, 10]
    L_add = L0 + (Lk - L0).sum(0)
    top3 = th.zeros_like(x).scatter_(1, idx[:, :3], v[:, :3])
    m2 = v[:, 1:2]
    cmean = -th.exp(th.distributions.Normal(0, 1).log_prob(m2)) / th.special.ndtr(m2).clamp(min=1e-12)
    xmean = th.where(th.zeros_like(x, dtype=th.bool).scatter_(1, idx[:, :2], True), x, cmean.expand_as(x))
    # additive around the mean background: reference = top two + every low input at its conditional mean
    Lm = forward(params, xmean)
    xs2 = xmean.unsqueeze(0).repeat(8, 1, 1)
    for k in range(8):
        xs2[k].scatter_(1, low_idx[:, k:k + 1], v[:, 2 + k:3 + k])
    L_addm = Lm + (forward(params, xs2) - Lm).sum(0)
    c_full = L_full.argmax(-1) == p2
    out = {}
    for nm, L in [("additive", L_add), ("additive_meanref", L_addm), ("Z2", L0), ("Z3", forward(params, top3)), ("mean_bg", forward(params, xmean))]:
        c = L.argmax(-1) == p2
        out[nm] = (c.float().mean().item(), (c == c_full).float().mean().item())
    out["full"] = (c_full.float().mean().item(), 1.0)
    # how good is the additive surrogate at the logit level: R^2 of logit margins (target minus max other)
    def margin(L):
        lt = L.gather(1, p2[:, None])[:, 0]
        Lo = L.scatter(1, p2[:, None], -th.inf)
        return lt - Lo.max(-1).values
    mf, ma = margin(L_full), margin(L_add)
    out["margin_R2"] = (1 - ((mf - ma) ** 2).mean() / mf.var()).item()
    return out


names, acc, models = load_bench(("mult",))
rows = []
base = run(example())
print("example:", {k: np.round(v, 4) for k, v in base.items()})
for i, (nm, a, m) in enumerate(zip(names, acc, models)):
    o = run(m, seed=1 + i)
    rows.append([a] + [o[k][0] for k in ("full", "additive", "Z2", "Z3", "mean_bg", "additive_meanref")] + [o["additive"][1], o["margin_R2"], o["additive_meanref"][1]])
rows = np.array(rows)
sig = np.array([float(str(n).split("_")[0][4:]) for n in names])
print(f"\nN = {N} inputs per copy; columns: RMSE against true accuracy, bias, correlation")
for j, nm in enumerate(["full (sampling check)", "additive", "Z2", "Z3", "mean_bg", "additive_meanref"]):
    e = rows[:, 1 + j]
    parts = [f"{nm:22s} all: RMSE {np.sqrt(np.mean((e - rows[:, 0])**2)):.4f} bias {np.mean(e - rows[:, 0]):+.4f} corr {np.corrcoef(e, rows[:, 0])[0, 1]:.3f}"]
    for s in (0.005, 0.01, 0.02):
        k = sig == s
        parts.append(f"σ={s}: RMSE {np.sqrt(np.mean((e[k] - rows[k, 0])**2)):.4f} bias {np.mean(e[k] - rows[k, 0]):+.4f}")
    print(" | ".join(parts))
print(f"additive per-input agreement with network: mean {rows[:, 7].mean():.4f} (min {rows[:, 7].min():.4f}); margin R^2 mean {rows[:, 8].mean():.3f}; mean-reference agreement {rows[:, 9].mean():.4f}")
np.savez("/data/output/research/algzoo/20261016_run2/reports/additive_pilot.npz", names=names, rows=rows,
         cols=np.array(["true", "full", "additive", "Z2", "Z3", "mean_bg", "additive_meanref", "agree_add", "margin_R2", "agree_add_meanref"]))

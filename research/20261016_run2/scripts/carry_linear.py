"""Is the two-spike carry failure a property of the (nearly) linear free dynamics?

Deterministic θ grid per position pair (weights g(θ)Δθ), no sampling. For each grid input:
- exact: run the network; carry loss = right at the last spike's step T, wrong at the end.
- frozen: from step T+1 on, replace ReLU(W_hh h) by D W_hh h with D the activation pattern of step T+1.
- frozen2: D fixed per step as in the base model's own two-spike trajectory? (not used) -> instead D = pattern at T+2.
Reports carry loss per model under each, and the correlation across the 96 copies.
"""
import numpy as np
import torch as th

from algzoo.bench import load_bench
from algzoo.core import example
from algzoo.exact_z2 import HI, LO, g_angle

dev = "cuda"
NG = 2000
th_grid = LO + (np.arange(NG) + 0.5) * (HI - LO) / NG
wgrid = g_angle(th_grid) * (HI - LO) / NG          # per-pair mass of each grid angle
pairs = [(a, b) for a in range(10) for b in range(10) if a != b]


@th.no_grad()
def carry(params, freeze_at=1):
    w_hi, w_hh, w_oh = (th.as_tensor(p, device=dev, dtype=th.float64) for p in params)
    out = {"exact": 0.0, "frozen": 0.0, "write_right": 0.0, "z2": 0.0, "frozen_z2": 0.0}
    c, s = th.as_tensor(np.cos(th_grid), device=dev), th.as_tensor(np.sin(th_grid), device=dev)
    W = th.as_tensor(wgrid, device=dev)
    for p1, p2 in pairs:
        T = max(p1, p2)
        h = th.zeros(NG, 16, device=dev, dtype=th.float64)
        for t in range(T + 1):
            x = c if t == p1 else (s if t == p2 else th.zeros_like(c))
            h = th.relu(h @ w_hh.T + x[:, None] * w_hi)
        right_T = (h @ w_oh.T).argmax(-1) == p2 + 9 - T
        he, hf = h.clone(), h.clone()
        D = None
        for t in range(T + 1, 10):
            he = th.relu(he @ w_hh.T)
            pre = hf @ w_hh.T
            if t == T + freeze_at or D is None and t > T + freeze_at - 1:
                D = (pre > 0).double()
            hf = pre * D if D is not None else th.relu(pre)
        ok_e = (he @ w_oh.T).argmax(-1) == p2
        ok_f = (hf @ w_oh.T).argmax(-1) == p2
        out["write_right"] += (W * right_T).sum().item()
        out["exact"] += (W * (right_T & ~ok_e)).sum().item()
        out["frozen"] += (W * (right_T & ~ok_f)).sum().item()
        out["z2"] += (W * ok_e).sum().item()
        out["frozen_z2"] += (W * ok_f).sum().item()
    return out


names, acc, models = load_bench(("mult",))
for fa in (1, 2):
    b = carry(example(), fa)
    rows = np.array([[carry(m, fa)[k] for k in ("exact", "frozen", "z2", "frozen_z2")] for m in models])
    print(f"freeze pattern of step T+{fa}: base carry loss exact {b['exact']:.4f} frozen {b['frozen']:.4f}; base Z2 grid {b['z2']:.4f}")
    print(f"  copies: carry loss exact mean {rows[:,0].mean():.4f}, frozen mean {rows[:,1].mean():.4f}, corr {np.corrcoef(rows[:,0], rows[:,1])[0,1]:.3f},"
          f" RMSE {np.sqrt(np.mean((rows[:,0]-rows[:,1])**2)):.4f}; Z2 corr {np.corrcoef(rows[:,2], rows[:,3])[0,1]:.3f}")
np.save("/data/output/research/algzoo/20261016_run2/reports/carry_linear.npy", rows)

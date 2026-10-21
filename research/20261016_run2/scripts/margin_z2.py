"""Diagnostic: margin-thresholded Z_2. Count a two-spike input as right only if its final logit margin
(target minus best other) exceeds δ·r, r = |(m1, m2)| (homogeneous). δ is set on the base model alone
(its true accuracy 0.9531); the RMSE over copies is then honest. Also shows the best δ in hindsight (not allowed,
diagnostic upper bound). Deterministic θ grid, no sampling.
"""
import numpy as np
import torch as th

from algzoo.bench import load_bench
from algzoo.core import example
from algzoo.exact_z2 import HI, LO, g_angle

dev = "cuda"
NG = 2000
tg = LO + (np.arange(NG) + 0.5) * (HI - LO) / NG
W = th.as_tensor(g_angle(tg) * (HI - LO) / NG, device=dev)
cs, sn = th.as_tensor(np.cos(tg), device=dev), th.as_tensor(np.sin(tg), device=dev)
pairs = [(a, b) for a in range(10) for b in range(10) if a != b]
deltas = np.linspace(0, 3, 61)


@th.no_grad()
def curve(params):
    w_hi, w_hh, w_oh = (th.as_tensor(p, device=dev, dtype=th.float64) for p in params)
    out = np.zeros(len(deltas))
    for p1, p2 in pairs:
        h = th.zeros(NG, 16, device=dev, dtype=th.float64)
        for t in range(10):
            x = cs if t == p1 else (sn if t == p2 else th.zeros_like(cs))
            h = th.relu(h @ w_hh.T + x[:, None] * w_hi)
        L = h @ w_oh.T
        Lo = L.clone(); Lo[:, p2] = -th.inf
        m = (L[:, p2] - Lo.max(-1).values).cpu().numpy()     # r = 1 on the grid (unit (cos, sin))
        out += np.array([(W.cpu().numpy() * (m > d)).sum() for d in deltas])
    return out


names, acc, models = load_bench(("mult",))
cb = curve(example())
d_base = deltas[np.argmin(abs(cb - 0.9531))]
print("base curve:", dict(zip(deltas[::10].round(2), cb[::10].round(4))), " δ from base:", d_base)
C = np.array([curve(m) for m in models])
sig = np.array([float(str(n).split("_")[0][4:]) for n in names])
for lab, j in [("δ = 0 (Z_2)", 0), (f"δ = {d_base:.2f} (set on base)", int(np.argmin(abs(cb - 0.9531))))]:
    e = C[:, j]
    print(f"{lab}: RMSE {np.sqrt(np.mean((e - acc)**2)):.4f} bias {np.mean(e - acc):+.4f} corr {np.corrcoef(e, acc)[0,1]:.3f};"
          + " ".join(f" σ={s}: {np.sqrt(np.mean((e[sig==s]-acc[sig==s])**2)):.4f}" for s in (0.005, 0.01, 0.02)))
rm = np.sqrt(((C - acc[:, None]) ** 2).mean(0))
print(f"hindsight best δ {deltas[rm.argmin()]:.2f}: RMSE {rm.min():.4f} (diagnostic only)")

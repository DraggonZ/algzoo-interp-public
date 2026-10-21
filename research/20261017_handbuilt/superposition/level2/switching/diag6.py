"""Development check: in single nodes (fixed positions of the largest and second-largest, values in narrow bins),
what do the nine margins (second-largest's score minus each other score) look like? Mean, SD, skew, the sampled share
below zero and the Gaussian's share below zero."""
import numpy as np, torch
from scipy.stats import norm
from algzoo.core import example
DEV = "cuda"; dt = torch.float64
wi, wh, wo = [torch.as_tensor(np.asarray(w, float), device=DEV, dtype=dt) for w in example()]
X = torch.randn(4_000_000, 10, device=DEV, dtype=dt, generator=torch.Generator(device=DEV).manual_seed(5))
h = torch.zeros(len(X), 16, device=DEV, dtype=dt)
for t in range(10): h = torch.relu(h @ wh.T + X[:, t:t + 1] * wi)
sc = h @ wo.T; o = X.argsort(1, descending=True); pL, pS = o[:, 0], o[:, 1]
L = X.gather(1, pL[:, None])[:, 0]; S = X.gather(1, pS[:, None])[:, 0]
for (a, b, l0, s0) in [(3, 6, 1.9, 1.3), (7, 2, 1.9, 1.3), (5, 8, 1.6, 1.4)]:
    m = (pL == a) & (pS == b) & ((L - l0).abs() < 0.05) & ((S - s0).abs() < 0.05)
    d = (sc[m, b:b + 1] - sc[m]).cpu().numpy(); print(f"node largest at {a} ({l0}), second at {b} ({s0}): {m.sum().item()} lists, right {100*(d[:, [k for k in range(10) if k != b]] > 0).all(1).mean():.1f}%")
    for k in range(10):
        if k == b: continue
        x = d[:, k]; mu, sd = x.mean(), x.std(); sk = ((x - mu) ** 3).mean() / sd ** 3
        print(f"   vs position {k}: mean {mu:7.2f} SD {sd:6.2f} skew {sk:+.2f} | below zero: sampled {100*(x<0).mean():5.2f}%, Gaussian {100*norm.cdf(-mu/sd):5.2f}%")

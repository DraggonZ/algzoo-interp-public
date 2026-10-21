"""Pre-registered diagnostic for the new seeds (not used to choose anything): each model's sampled three-input reduction
(others at the constant -0.46, as in the estimate, and at their conditional mean), against its truth and its estimate.
Splits each estimate error into reduction error (reduction minus truth) and band error (estimate minus reduction)."""
import numpy as np
import torch as th

from algzoo.core import load

dev = "cuda"
N = 2 ** 23
nd = th.distributions.Normal(0.0, 1.0)
g = th.Generator(device=dev).manual_seed(777)
X = th.randn(N, 10, device=dev, generator=g)
o = X.argsort(1); tgt = o[:, -2]
m3 = X.gather(1, o[:, -3:-2])[:, 0]
cm = -th.exp(nd.log_prob(m3)) / nd.cdf(m3).clamp_min(1e-12)
red = {}
for lab, bg in [("const", th.full_like(m3, -0.46)), ("cond", cm)]:
    Xr = bg[:, None].expand_as(X).clone(); Xr.scatter_(1, o[:, -3:], X.gather(1, o[:, -3:])); red[lab] = Xr
S = np.load("/data/output/research/algzoo/20261016_run3/reports/new_score.npy")
rows = []
for seed, a, est, *_ in S:
    p = load(f"/data/output/research/algzoo/20261016_run3/models/2nd_argmax_16_10_1073741824_{int(seed)}.pth")
    w_hi, w_hh, w_oh = (th.as_tensor(np.asarray(q), dtype=th.float32, device=dev) for q in p)
    acc = {}
    for lab, Xr in red.items():
        h = th.zeros(N, 16, device=dev)
        for t in range(10):
            h = th.relu(h @ w_hh.T + Xr[:, t, None] * w_hi)
        acc[lab] = ((h @ w_oh.T).argmax(1) == tgt).float().mean().item()
    rows.append((a, est, acc["const"], acc["cond"]))
    print(f"seed {int(seed)}: estimate error {100 * (est - a):+.2f} = reduction error {100 * (acc['const'] - a):+.2f} + band error {100 * (est - acc['const']):+.2f} "
          f"(conditional-mean reduction error {100 * (acc['cond'] - a):+.2f})", flush=True)
R = np.array(rows)
print(f"RMSE over 16: reduction (constant) {100 * np.sqrt(np.mean((R[:, 2] - R[:, 0]) ** 2)):.2f}, reduction (conditional mean) {100 * np.sqrt(np.mean((R[:, 3] - R[:, 0]) ** 2)):.2f}, "
      f"band step against its own target {100 * np.sqrt(np.mean((R[:, 1] - R[:, 2]) ** 2)):.2f} points")

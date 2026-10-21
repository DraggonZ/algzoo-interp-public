"""r6a analysis 2: why the gap in mem6 costs 4 points on lists whose second is not at position 1. Hypothesis: mem6 is
switched off (clipped at 0) on some lists, so the gap it reads is not added there while the readout still takes it off.
Reads, 1M lists (seed 1000), tap-5 ring: share of lists with mem6 off at step 9; mem6's pre-activation at step 9 in
units of the largest through step 8 (quantiles); accuracy split by whether mem6 is off."""
import numpy as np, torch
from algzoo.handbuilt_v6a import build_v6a
DEV, DT = "cuda", torch.float64
n = 1_000_000
X = torch.randn(n, 10, device=DEV, dtype=DT, generator=torch.Generator(device=DEV).manual_seed(1000))
Y2 = X.argsort(1, descending=True)[:, 1]
m8 = X[:, :9].max(1).values.clamp(min=0)
for g in (0.0, 1.0):
    net = build_v6a(ring_tap=5, gap_eff=g)
    wi, wh, wo = (torch.tensor(np.asarray(p, float), device=DEV, dtype=DT) for p in net.params)
    h = torch.zeros(n, 16, device=DEV, dtype=DT)
    for t in range(10):
        pre = h @ wh.T + X[:, t:t + 1] * wi; h = torch.relu(pre)
    k = net.idx["mem6"]; off = pre[:, k] <= 0; ok = (h @ wo.T).argmax(1) == Y2
    r = (pre[:, k] / m8.clamp(min=1e-3)).cpu().numpy()
    print(f"gap {g}: mem6 off at step 9 on {100 * off.double().mean().item():.2f}% of lists; "
          f"pre/m8 quantiles 0.1% {np.quantile(r, 0.001):.3f} 1% {np.quantile(r, 0.01):.3f} 10% {np.quantile(r, 0.1):.3f} 50% {np.quantile(r, 0.5):.3f}; "
          f"accuracy off {100 * ok[off].double().mean().item():.1f} on {100 * ok[~off].double().mean().item():.1f}; "
          f"answer at 1 when wrong {100 * ((h @ wo.T).argmax(1)[~ok] == 1).double().mean().item():.1f}%")
    # where x_1 is the largest of the first nine (its cell hides it)
    big1 = X[:, :9].argmax(1) == 1; rb = (pre[:, k][big1] / m8[big1].clamp(min=1e-3)).cpu().numpy()
    print(f"   largest at 1: pre/m8 quantiles 0.1% {np.quantile(rb, 0.001):.3f} 1% {np.quantile(rb, 0.01):.3f} 50% {np.quantile(rb, 0.5):.3f} 99% {np.quantile(rb, 0.99):.3f}")
    print(f"   lists with the largest at position 1: mem6 off {100 * off[big1].double().mean().item():.1f}%, others {100 * off[~big1].double().mean().item():.1f}%")

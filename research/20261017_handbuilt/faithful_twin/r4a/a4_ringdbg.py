"""r4a analysis 4: why the memory ring costs 14 points: score difference (ring minus no ring) per position.
With any argument: the ring with the readout compensation (ring_comp)."""
import sys, numpy as np, torch
RC = len(sys.argv) > 1
from algzoo.handbuilt_v4a import build_v4a
DEV, DT = "cuda", torch.float64
n = 200_000
X = torch.randn(n, 10, device=DEV, dtype=DT, generator=torch.Generator(device=DEV).manual_seed(5))
def sc(net):
    wi, wh, wo = (torch.tensor(np.asarray(p, float), device=DEV, dtype=DT) for p in net.params)
    h = torch.zeros(n, len(wi), device=DEV, dtype=DT); Hs = []
    for t in range(10):
        h = torch.relu(h @ wh.T + X[:, t:t + 1] * wi); Hs.append(h)
    return h @ wo.T, torch.stack(Hs, 1)
a, Ha = sc(build_v4a(bw_late=True)); b, Hb = sc(build_v4a(bw_late=True, ring_w=0.07, ring_comp=RC))
d = (b - a).abs()
print("median |score change| per position:", " ".join(f"{p}:{d[:, p].median():.3g}" for p in range(10)))
print("90th pct:", " ".join(f"{p}:{d[:, p].quantile(0.9):.3g}" for p in range(10)))
net = build_v4a(bw_late=True, ring_w=0.07)
dh = (Hb - Ha).abs().median(0).values
for k, i in net.idx.items(): print(k, " ".join(f"{v:.2g}" for v in dh[:, i].tolist()))

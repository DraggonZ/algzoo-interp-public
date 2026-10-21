"""r4a analysis 5: the memory ring (w 0.07) with the readout adding back the current largest's looped backward wave.
Own accuracy on 1M lists (seed 1000), with and without the late gate writing the backward wave."""
import numpy as np, torch
from algzoo.handbuilt_v4a import build_v4a
DEV, DT = "cuda", torch.float64
n = 1_000_000
X = torch.randn(n, 10, device=DEV, dtype=DT, generator=torch.Generator(device=DEV).manual_seed(1000))
Y2 = X.argsort(1, descending=True)[:, 1]
def acc(net):
    wi, wh, wo = (torch.tensor(np.asarray(p, float), device=DEV, dtype=DT) for p in net.params)
    h = torch.zeros(n, len(wi), device=DEV, dtype=DT)
    for t in range(10): h = torch.relu(h @ wh.T + X[:, t:t + 1] * wi)
    return 100 * ((h @ wo.T).argmax(1) == Y2).double().mean().item()
for kw in (dict(ring_w=0.07, ring_comp=True), dict(bw_late=True, ring_w=0.07, ring_comp=True),
           dict(bw_late=True, ring_w=0.07, ring_comp=True, eps=0.0), dict(bw_late=True, ring_w=0.062, ring_comp=True)):
    print(kw, f"{acc(build_v4a(**kw)):.2f}")

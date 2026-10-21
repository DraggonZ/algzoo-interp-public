"""r5a analysis 2: does a faster-growing hiding wave make the memory ring cheaper?
The ring's leftover cost (r4a-d) comes partly from backward-wave writes g(-k) d into cells that hold no position yet;
g(-k) = K rho^-k cos(phi - k theta) / cos(phi) shrinks as rho grows. Scale rules kept from v4a: the clock gain gc grows
with rho^8 (gc / rho^8 = 1e10 / 4^8, so the clocks still clip only within the same fraction of a jump), and the linear
route's level lf with |g(8)| (lf / |g(8)| as at rho 4). Own accuracy on 1M lists (seed 1000), no ring and ring 0.062
with the readout compensation, late gate writing the backward wave."""
import numpy as np, torch
from algzoo.handbuilt_v4a import build_v4a, wave_2d, g_of
DEV, DT = "cuda", torch.float64
n = 1_000_000
X = torch.randn(n, 10, device=DEV, dtype=DT, generator=torch.Generator(device=DEV).manual_seed(1000))
Y2 = X.argsort(1, descending=True)[:, 1]


def acc(net):
    wi, wh, wo = (torch.tensor(np.asarray(p, float), device=DEV, dtype=DT) for p in net.params)
    h = torch.zeros(n, len(wi), device=DEV, dtype=DT)
    for t in range(10): h = torch.relu(h @ wh.T + X[:, t:t + 1] * wi)
    return 100 * ((h @ wo.T).argmax(1) == Y2).double().mean().item()


A4 = wave_2d(rho_w=4.0); g8_4 = abs(g_of(*A4, 8))
for rho in (4.0, 6.0, 8.0):
    A = wave_2d(rho_w=rho); g8 = abs(g_of(*A, 8))
    gc = 1e10 * (rho / 4) ** 8; lf = 2e5 * g8 / g8_4
    back = " ".join(f"{g_of(*A, -k):.3f}" for k in range(1, 5))
    print(f"rho {rho:.0f}: g(-1..-4) {back}  |g(8)| {g8:.3g}  gc {gc:.3g}  lf {lf:.3g}")
    for ring in (0.0, 0.062):
        kw = dict(bw_late=True, rho_w=rho, gc=gc, lf=lf, ring_w=ring, ring_comp=bool(ring))
        print(f"   ring {ring}: accuracy {acc(build_v4a(**kw)):.2f}")

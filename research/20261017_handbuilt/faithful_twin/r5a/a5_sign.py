"""r5a analysis 5: a ring with a negative loop weight.
a4_last: the ring carries about -0.05 x relu(x_0) into position 8 at the last step (position 0's cell). Position 0 is
scored (1 - delta) x the runner-up, delta 0.01, so when the second is at 8 any loss over 1% of its value hands the
answer to position 0. A negative loop weight (w = -0.062) keeps the ring's size (modes at 22.5, 67.5, 112.5, 157.5
degrees, modulus 0.70) and turns that loss into a gain: the true second's score rises instead, and a non-second
position 8 wins only if it is within about 5% of x_0 below the second. Same for the empty-cell writes reaching
position 7. Own accuracy on 1M lists (seed 1000), by position of the second; ring modes of the memory block."""
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
    ok = (h @ wo.T).argmax(1) == Y2
    return 100 * ok.double().mean().item(), [100 * ok[Y2 == q].double().mean().item() for q in range(10)]


for rho in (4.0, 6.0):
    gc = 1e10 * (rho / 4) ** 8; lf = 2e5 * abs(g_of(*wave_2d(rho_w=rho), 8)) / abs(g_of(*wave_2d(rho_w=4.0), 8))
    for w in (0.062, -0.062):
        net = build_v4a(bw_late=True, rho_w=rho, gc=gc, lf=lf, ring_w=w, ring_comp=True)
        a, by = acc(net)
        M = [net.idx["linear"]] + [net.idx[f"mem{k}"] for k in range(7)]
        ev = np.linalg.eigvals(net.params[1][np.ix_(M, M)])
        modes = sorted({(round(float(np.degrees(abs(np.angle(e)))), 1), round(float(abs(e)), 3)) for e in ev if abs(e) > 0.1})
        print(f"rho {rho:.0f} w {w:+.3f}: {a:.2f}  by position of the second: " + " ".join(f"{v:.1f}" for v in by))
        print(f"   memory-block modes (deg, modulus): {modes}")

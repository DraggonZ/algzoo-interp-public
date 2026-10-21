"""r5a analysis 3: where does the ring still cost accuracy? (rho 4 and rho 6 builds, ring 0.062 with readout
compensation, late gate writing the backward wave.) Own accuracy on 1M lists (seed 1000) with two parts of the ring's
traffic removed by hand inside the forward pass (analysis only, not a network):
  (i)  at the last step the ring carries position 0's cell into the linear route (position 8): removed;
  (ii) backward-wave writes by the late gate into cells that hold no position yet (cell k at step t holds x_{t-2-k};
       none if k >= t - 1): removed;
  (iii) both. Errors also split by where the true second is."""
import numpy as np, torch
from algzoo.handbuilt_v4a import build_v4a, wave_2d, g_of
DEV, DT = "cuda", torch.float64
n = 1_000_000
X = torch.randn(n, 10, device=DEV, dtype=DT, generator=torch.Generator(device=DEV).manual_seed(1000))
Y2 = X.argsort(1, descending=True)[:, 1]
W_RING, RHO_MEM, U = 0.062, 0.99, 1.0


def run(net, cut_last=False, cut_pre=False):
    wi, wh, wo = (torch.tensor(np.asarray(p, float), device=DEV, dtype=DT) for p in net.params)
    ix = net.idx; L, S, LG = ix["linear"], ix["store"], ix["late"]; Y = [ix[f"mem{k}"] for k in range(7)]
    h = torch.zeros(n, len(wi), device=DEV, dtype=DT)
    for t in range(10):
        p = h @ wh.T + X[:, t:t + 1] * wi
        if cut_last and t == 9:
            p[:, L] -= W_RING * h[:, Y[6]] - W_RING * U * RHO_MEM ** 6 / 20.0 * h[:, S]
        if cut_pre:
            for k in range(1, 7):
                if k >= t - 1:
                    p[:, Y[k]] -= wh[Y[k], LG] * h[:, LG]
        h = torch.relu(p)
    pred = (h @ wo.T).argmax(1); ok = pred == Y2
    by = [100 * ok[Y2 == q].double().mean().item() for q in range(10)]
    return 100 * ok.double().mean().item(), by


for rho in (4.0, 6.0):
    gc = 1e10 * (rho / 4) ** 8; lf = 2e5 * abs(g_of(*wave_2d(rho_w=rho), 8)) / abs(g_of(*wave_2d(rho_w=4.0), 8))
    for ring in (0.0, W_RING):
        net = build_v4a(bw_late=True, rho_w=rho, gc=gc, lf=lf, ring_w=ring, ring_comp=bool(ring))
        for cl, cp in ((False, False), (True, False), (False, True), (True, True)):
            if not ring and (cl or cp): continue
            a, by = run(net, cl, cp)
            print(f"rho {rho:.0f} ring {ring}: cut last-step loop {cl!s:5s} cut writes into empty cells {cp!s:5s}: "
                  f"{a:.2f}   by position of the second: " + " ".join(f"{v:.1f}" for v in by))

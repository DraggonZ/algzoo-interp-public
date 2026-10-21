"""r5a analysis 4: what does the ring carry into position 8 at the last step? (rho 6 build, ring 0.062.)
Loop term at step 9: w x (cell 6 at step 8) - w U rho^6 x store/s (the store part removes the looped shared level).
Least squares on x_0, the largest m, x_0 x [x_0 > 0] and the readout's compensation for the current largest; the
loop's typical size against the gap between the second and the third."""
import numpy as np, torch
from algzoo.handbuilt_v4a import build_v4a, wave_2d, g_of
DEV, DT = "cuda", torch.float64
n = 400_000
X = torch.randn(n, 10, device=DEV, dtype=DT, generator=torch.Generator(device=DEV).manual_seed(1000))
rho = 6.0; gc = 1e10 * (rho / 4) ** 8; lf = 2e5 * abs(g_of(*wave_2d(rho_w=rho), 8)) / abs(g_of(*wave_2d(rho_w=4.0), 8))
net = build_v4a(bw_late=True, rho_w=rho, gc=gc, lf=lf, ring_w=0.062, ring_comp=True)
wi, wh, wo = (torch.tensor(np.asarray(p, float), device=DEV, dtype=DT) for p in net.params)
ix = net.idx; S, Y6, L = ix["store"], ix["mem6"], ix["linear"]
h = torch.zeros(n, len(wi), device=DEV, dtype=DT); H = []
for t in range(10):
    h = torch.relu(h @ wh.T + X[:, t:t + 1] * wi); H.append(h)
loop = (0.062 * H[8][:, Y6] - 0.062 * 0.99 ** 6 / 20.0 * H[8][:, S]).cpu().numpy()
x = X.cpu().numpy(); m = np.maximum(x.max(1), 0); x0 = x[:, 0]
v = -np.sort(-x, 1); gap23 = v[:, 1] - v[:, 2]
print(f"loop term at the last step: median |.| {np.median(np.abs(loop)):.4f}, 90th pct {np.percentile(np.abs(loop), 90):.4f};"
      f" median gap second-third {np.median(gap23):.4f}")
A = np.c_[x0, m, x0 * (x0 > 0), np.ones(n)]
c, *_ = np.linalg.lstsq(A, loop, rcond=None); r2 = 1 - ((A @ c - loop) ** 2).mean() / loop.var()
print(f"fit loop = {c[0]:+.4f} x0 {c[1]:+.4f} m {c[2]:+.4f} x0[x0>0] {c[3]:+.4f}, R2 {r2:.3f}")
for name, mask in (("x0 the largest", x.argmax(1) == 0), ("x0 > 0, not the largest", (x0 > 0) & (x.argmax(1) != 0)),
                   ("x0 < 0", x0 < 0)):
    print(f"  {name:26s} share {mask.mean():.2f}: median loop {np.median(loop[mask]):+.4f}, median |loop| {np.median(np.abs(loop[mask])):.4f}")

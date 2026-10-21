"""Spread of 'mistakes reproduced' across copies with ARC's weight noise (SD 0.0025) and SD 0.001: 20 copies each,
grid 40 x 48 x 48. Mean, range and middle 80% (all mistakes and clear mistakes)."""
import numpy as np, torch
import store_ladder as sl
DEV = "cuda"; dt = torch.float64; T = sl.W
def ans(W, X):
    h = torch.zeros(len(X), 16, device=DEV, dtype=dt)
    for t in range(10): h = torch.relu(h @ W[1].T + X[:, t:t + 1] * W[0])
    return (h @ W[2].T).argmax(1)
g = torch.Generator(device=DEV).manual_seed(99)
for s in (0.001, 0.0025):
    Ws = [[w + s * torch.randn(w.shape, device=DEV, dtype=dt, generator=g) for w in T] for _ in range(20)]
    num = np.zeros(20); numc = np.zeros(20); den = denc = 0.0; nV = len(sl.wtt)
    for i in range(0, 720, 16):
        tt = sl.full_mean.T[i:i + 16]; pos = tt.repeat_interleave(nV, 0); val = sl.Vt.repeat(len(tt), 1); n = len(pos)
        bmu = -torch.exp(-val[:, 2] ** 2 / 2) / np.sqrt(2 * np.pi) / (0.5 * (1 + torch.erf(val[:, 2] / np.sqrt(2))))
        X = bmu[:, None].repeat(1, 10); ar = torch.arange(n, device=DEV)
        for k in range(3): X[ar, pos[:, k]] = val[:, k]
        w = sl.wtt.repeat(len(tt)); y = pos[:, 1]; S0 = sl.run(X); a0 = S0.argmax(1); wr = a0 != y
        clear = wr & ((S0[ar, a0] - S0[ar, y]) > 2.41)
        den += (w * wr).sum().item(); denc += (w * clear).sum().item()
        for c, W in enumerate(Ws):
            same = ans(W, X) == a0; num[c] += (w * (wr & same)).sum().item(); numc[c] += (w * (clear & same)).sum().item()
    m, mc = 100 * num / den, 100 * numc / denc
    print(f"SD {s}: mistakes reproduced mean {m.mean():.1f}% (range {m.min():.1f}-{m.max():.1f}, middle 80% {np.quantile(m,.1):.1f}-{np.quantile(m,.9):.1f}); clear mistakes mean {mc.mean():.1f}% (range {mc.min():.1f}-{mc.max():.1f})")

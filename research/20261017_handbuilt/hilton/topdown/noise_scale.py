"""A reference scale for 'mistakes reproduced': copies of the trained network with every weight + N(0, s^2) noise
(ARC's noise is s = 0.0025), scored like the explanations: on the grid (40 x 48 x 48), the share of the trained
network's mistakes each copy makes the same way (all, coin flips, middle, clear), mean over 6 copies per s."""
import numpy as np, torch
import store_ladder as sl
DEV = "cuda"; dt = torch.float64
T = sl.W
def net(W):
    def f(X):
        h = torch.zeros(len(X), 16, device=DEV, dtype=dt)
        for t in range(10): h = torch.relu(h @ W[1].T + X[:, t:t + 1] * W[0])
        return h @ W[2].T
    return f
g = torch.Generator(device=DEV).manual_seed(12)
levels = [0.0005, 0.001, 0.0025, 0.005]
copies = {s: [net([w + s * torch.randn(w.shape, device=DEV, dtype=dt, generator=g) for w in T]) for _ in range(6)] for s in levels}
nV = len(sl.wtt); rec = {s: [[] for _ in range(6)] for s in levels}; acc = {s: np.zeros(6) for s in levels}; M, Wt = [], []; Z = 0.0
for i in range(0, 720, 16):
    tt = sl.full_mean.T[i:i + 16]; pos = tt.repeat_interleave(nV, 0); val = sl.Vt.repeat(len(tt), 1); n = len(pos)
    bmu = -torch.exp(-val[:, 2] ** 2 / 2) / np.sqrt(2 * np.pi) / (0.5 * (1 + torch.erf(val[:, 2] / np.sqrt(2))))
    X = bmu[:, None].repeat(1, 10); ar = torch.arange(n, device=DEV)
    for k in range(3): X[ar, pos[:, k]] = val[:, k]
    w = sl.wtt.repeat(len(tt)); y = pos[:, 1]; S0 = sl.run(X); a0 = S0.argmax(1); wr = a0 != y; Z += w.sum().item()
    M.append((S0[ar, a0] - S0[ar, y])[wr].cpu().numpy()); Wt.append(w[wr].cpu().numpy())
    for s in levels:
        for c, f in enumerate(copies[s]):
            a1 = f(X).argmax(1); rec[s][c].append((a1 == a0)[wr].cpu().numpy()); acc[s][c] += (w * (a1 == y)).sum().item()
M, Wt = np.concatenate(M), np.concatenate(Wt); band = np.digitize(M, [0.89, 2.41])
print(f"{'noise SD per weight':22s} {'accuracy':>9s} {'all':>7s} {'coin flips':>11s} {'middle':>8s} {'clear':>8s}")
for s in levels:
    R = [np.concatenate(rec[s][c]) for c in range(6)]
    def sc(mask): return np.mean([100 * (Wt[mask] * r[mask]).sum() / Wt[mask].sum() for r in R])
    print(f"{s:<22.4f} {100*np.mean(acc[s])/Z:8.2f}% {sc(np.ones_like(band, bool)):6.1f}% {sc(band==0):10.1f}% {sc(band==1):7.1f}% {sc(band==2):7.1f}%")

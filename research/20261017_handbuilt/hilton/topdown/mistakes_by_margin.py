"""Mistakes reproduced, split by how clearly the trained network's wrong answer wins: its score minus the right
answer's score, in three bands (lowest, middle and highest third of the mistakes' probability). Grid 40 x 48 x 48."""
import numpy as np, torch
import store_ladder as sl, store_ladder2 as s2, detector_ladder as dl
DEV = "cuda"; dt = torch.float64
wi, wh, wo = sl.W
d = np.load("../../bridge/v18ts_matched2.npz"); H = [torch.tensor(d[k], device=DEV, dtype=dt) for k in ("wi", "wh", "wo")]
def net(W):
    def f(X):
        h = torch.zeros(len(X), 16, device=DEV, dtype=dt)
        for t in range(10): h = torch.relu(h @ W[1].T + X[:, t:t + 1] * W[0])
        return h @ W[2].T
    return f
Wk = wh.clone(); Wk[2, 5] = 0
models = [("first twin", net(H)), ("store as 17.6 x the largest", lambda X: sl.run(X, rule=lambda L, a: 17.6 * L)),
          ("store as age table", lambda X: sl.run(X, rule=lambda L, a: sl.r[a.clamp(max=9)] * L)),
          ("store as age table + fitted correction", s2.run4), ("detector own weights + correction", lambda X: dl.run(X, dl.d4)),
          ("store without the keep gate's weight (-0.01)", net([wi, Wk, wo]))]
nV = len(sl.wtt); rec = {k: [] for k, _ in models}; M, Wt = [], []
for i in range(0, 720, 16):
    tt = sl.full_mean.T[i:i + 16]; pos = tt.repeat_interleave(nV, 0); val = sl.Vt.repeat(len(tt), 1); n = len(pos)
    bmu = -torch.exp(-val[:, 2] ** 2 / 2) / np.sqrt(2 * np.pi) / (0.5 * (1 + torch.erf(val[:, 2] / np.sqrt(2))))
    X = bmu[:, None].repeat(1, 10); ar = torch.arange(n, device=DEV)
    for k in range(3): X[ar, pos[:, k]] = val[:, k]
    w = sl.wtt.repeat(len(tt)); y = pos[:, 1]; S0 = sl.run(X); a0 = S0.argmax(1); wr = a0 != y
    M.append((S0[ar, a0] - S0[ar, y])[wr].cpu().numpy()); Wt.append(w[wr].cpu().numpy())
    for k, f in models: rec[k].append((f(X).argmax(1) == a0)[wr].cpu().numpy())
M, Wt = np.concatenate(M), np.concatenate(Wt); o = np.argsort(M); cw = np.cumsum(Wt[o]) / Wt.sum()
edges = [M[o][np.searchsorted(cw, q)] for q in (1 / 3, 2 / 3)]
band = np.digitize(M, edges)
print(f"wrong answer's lead over the right answer, band edges: {edges[0]:.2f}, {edges[1]:.2f} (median lead of the winner over the runner-up on all lists is about 8)")
print(f"{'':48s} {'lowest third':>13s} {'middle third':>13s} {'highest third':>14s}")
for k, _ in models:
    r = np.concatenate(rec[k]); print(f"{k:48s} " + " ".join(f"{100*(Wt[band==b]*r[band==b]).sum()/Wt[band==b].sum():12.1f}%" for b in range(3)))

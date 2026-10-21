"""Which incoming connections do the store (2) and detector (4) need? Keep only some of their incoming recurrent weights
(real values), zero the rest, everything else intact. Scored on the grid (40 x 48 x 48, no sampling) by mistakes
reproduced, agreement and accuracy."""
import numpy as np, torch
import store_ladder as sl
DEV = "cuda"; dt = torch.float64
wi, wh, wo = sl.W
G = {"S": [2], "D": [4], "C": [1, 6, 7, 8], "M": [0, 3, 10, 11, 12, 13, 14, 15], "K": [5], "L": [9]}
def masked(rows_keep):
    W = wh.clone()
    for r, keep in rows_keep.items():
        cols = [j for g in keep for j in G[g]]
        m = torch.ones(16, dtype=torch.bool, device=DEV); m[cols] = False; W[r, m] = 0
    return W
variants = [("store reads all but the keep gate", {2: "SDCML"}), ("store reads all but the late gate", {2: "SDCMK"}), ("store reads store + detector", {2: "SD"}),
            ("store reads store + detector + cancelling", {2: "SDC"}),
            ("store reads store + detector + cancelling + memory", {2: "SDCM"}),
            ("detector reads store + detector", {4: "SD"}),
            ("detector reads store + detector + cancelling", {4: "SDC"}),
            ("both read store + detector + cancelling", {2: "SDC", 4: "SDC"}),
            ("both read store + detector + cancelling + memory", {2: "SDCM", 4: "SDCM"})]
def run(X, W):
    h = torch.zeros(len(X), 16, device=DEV, dtype=dt)
    for t in range(10): h = torch.relu(h @ W.T + X[:, t:t + 1] * wi)
    return (h @ wo.T).argmax(1)
nV = len(sl.wtt); res = {k: np.zeros(3) for k, _ in variants}; Zw = Z = 0.0; Ws = {k: masked(v) for k, v in variants}
for i in range(0, 720, 16):
    tt = sl.full_mean.T[i:i + 16]; pos = tt.repeat_interleave(nV, 0); val = sl.Vt.repeat(len(tt), 1); n = len(pos)
    bmu = -torch.exp(-val[:, 2] ** 2 / 2) / np.sqrt(2 * np.pi) / (0.5 * (1 + torch.erf(val[:, 2] / np.sqrt(2))))
    X = bmu[:, None].repeat(1, 10); ar = torch.arange(n, device=DEV)
    for k in range(3): X[ar, pos[:, k]] = val[:, k]
    w = sl.wtt.repeat(len(tt)); y = pos[:, 1]; a0 = run(X, wh); wr = a0 != y; Zw += (w * wr).sum().item(); Z += w.sum().item()
    for k, _ in variants:
        a1 = run(X, Ws[k]); res[k] += [(w * (a1 == a0)).sum().item(), (w * (wr & (a1 == a0))).sum().item(), (w * (a1 == y)).sum().item()]
for k, (ag, mw, ac) in res.items():
    print(f"{k:52s} mistakes reproduced {100*mw/Zw:5.1f}%   agreement {100*ag/Z:6.2f}%   accuracy {100*ac/Z:6.2f}%")

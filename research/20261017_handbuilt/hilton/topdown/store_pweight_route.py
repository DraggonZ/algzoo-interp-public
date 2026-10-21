"""Which neurons carry the store's weight on the previous largest P? The store's weight on P at step t equals the sum
over source neurons j of W[2, j] times j's weight on P at step t-1 (plus the store's own and the detector's terms).
At age 6, split lists by P/L below or above 0.5 and show the median contribution of each source group."""
import numpy as np, torch
exec(open("store_jacobian.py").read().split("S = torch.stack(S, 1)")[0].replace("S.append(h[:, 2].clone()); JS.append(J[:, :, 2].clone())", "S.append(h[:, 2].clone()); JS.append(J.clone())"))
Jall = torch.stack(JS, 1).cpu().numpy()       # [n, t, s, neuron]
Xn = X.cpu().numpy(); N = len(Xn); W = wh.cpu().numpy()
r = np.full(N, -1); rp = np.full(N, -1); m = np.zeros(N); R = np.zeros((N, 10), int); RP = np.zeros((N, 10), int)
for t in range(10):
    R[:, t] = r; RP[:, t] = rp; new = Xn[:, t] > m; rp = np.where(new, r, rp); m = np.where(new, Xn[:, t], m); r = np.where(new, t, r)
G = {"store itself (2)": [2], "detector (4)": [4], "cancelling (1,6,7,8)": [1, 6, 7, 8], "keep gate (5)": [5], "late gate (9)": [9], "memory (0,3,10-15)": [0, 3, 10, 11, 12, 13, 14, 15]}
ar = np.arange(N)
for a in (4, 6):
    for lo, hi in ((0.0, 0.5), (0.5, 1.0)):
        contrib = {g: [] for g in G}; tot = []
        for t in range(a, 10):
            idx = np.where((R[:, t] == t - a) & (RP[:, t] >= 0) & (Xn[ar, np.maximum(R[:, t], 0)] > 0.3))[0]
            ratio = Xn[idx, RP[idx, t]] / Xn[idx, t - a]; idx = idx[(ratio >= lo) & (ratio < hi)]
            prevJ = Jall[idx, t - 1, RP[idx, t], :]          # each neuron's weight on P after step t-1
            on = Jall[idx, t, RP[idx, t], 2] != 0
            for g, js in G.items(): contrib[g].append((prevJ[:, js] * W[2, js]).sum(1))
            tot.append(Jall[idx, t, RP[idx, t], 2])
        print(f"age {a}, P/L in [{lo},{hi}): store's weight on P median {np.median(np.concatenate(tot)):+.2f}; from " + ", ".join(f"{g} {np.median(np.concatenate(v)):+.2f}" for g, v in contrib.items()), flush=True)

"""How faithful is the hand-built network (v18, matched into the trained network's coordinates by exact relabelling
and rescaling) to ARC's trained network? Two measures, both computed without sampling lists, on the grid of the
three largest numbers with the other seven at their truncated mean (../propagate/full_mean.py, grid 40 x 48 x 48;
on the intact trained network this grid gives 95.23%, the converged value is 95.32%).

(1) Hilton's metric: start from the hand-built weights H and treat the gap to the trained weights T as noise: copies
    H + e, each weight's e normal with SD |T - H| for that weight (random sign and size). Predicted trained accuracy =
    mean accuracy over 12 such copies. Also: the gap with its true signs, scaled by s = 0.25, 0.5, 0.75, 1 (a straight
    path from H to T).
(2) Group patching: run both networks on the same inputs; after each step, overwrite one group of the trained
    network's neurons with the hand-built network's values. Accuracy lost = that group's unfaithfulness in points.
    Also the reverse: patch everything except one group."""
import sys
import numpy as np, torch
sys.path.insert(0, "../propagate")
import full_mean
DEV = "cuda"; dt = torch.float64
Tw = [torch.tensor(np.asarray(p, float), device=DEV, dtype=dt) for p in __import__("algzoo.core", fromlist=["example"]).example()]
G = {"store and detector (2, 4)": [2, 4], "cancelling (1, 6, 7, 8)": [1, 6, 7, 8], "keep gate (5)": [5], "late gate (9)": [9],
     "memory (0, 3, 10-15)": [0, 3, 10, 11, 12, 13, 14, 15]}
V, wt = full_mean.grid(40, 48); V = torch.tensor(V, device=DEV, dtype=dt); wt = torch.tensor(wt, device=DEV, dtype=dt)


def grid_acc(step):
    """step(h_T, h_H, x) -> (h_T, h_H): one update of the (possibly coupled) pair. Accuracy of the trained readout."""
    nV = len(wt); tot = torch.zeros((), device=DEV, dtype=dt)
    for i in range(0, 720, 16):
        tt = full_mean.T[i:i + 16]; pos = tt.repeat_interleave(nV, 0); val = V.repeat(len(tt), 1); n = len(pos)
        bmu = -torch.exp(-val[:, 2] ** 2 / 2) / np.sqrt(2 * np.pi) / (0.5 * (1 + torch.erf(val[:, 2] / np.sqrt(2))))
        X = bmu[:, None].repeat(1, 10); ar = torch.arange(n, device=DEV)
        for k in range(3): X[ar, pos[:, k]] = val[:, k]
        hT = torch.zeros(n, 16, device=DEV, dtype=dt); hH = torch.zeros(n, 16, device=DEV, dtype=dt)
        for t in range(10): hT, hH = step(hT, hH, X[:, t:t + 1])
        tot += (((hT @ Tw[2].T).argmax(1) == pos[:, 1]).to(dt) * wt.repeat(len(tt))).sum()
    return (tot / (wt.sum() * 720)).item()


def plain(W):
    return lambda hT, hH, x: (torch.relu(hT @ W[1].T + x * W[0]), hH)


for f in sys.argv[1:] or ("v18ts_matched2.npz", "v18g50_matched.npz"):
    d = np.load(f if "/" in f else f"../../bridge/{f}"); H = [torch.tensor(d[k], device=DEV, dtype=dt) for k in ("wi", "wh", "wo")]
    print(f"== hand-built {f}", flush=True)
    a_T = grid_acc(plain(Tw))
    a_H = grid_acc(lambda hT, hH, x: (torch.relu(hT @ H[1].T + x * H[0]), hH)) if False else None
    # the hand-built network with its own readout: run it in the hT slot with its weights, readout via Tw would be wrong, so compute separately
    def own_acc(W):
        global Tw
        keep = Tw[2]; Tw[2] = W[2]; a = grid_acc(plain(W)); Tw[2] = keep; return a
    a_H = own_acc(H)
    print(f"   trained network {100*a_T:.2f}%   hand-built network on its own {100*a_H:.2f}%")
    g = torch.Generator(device=DEV).manual_seed(1); acc = []
    for c in range(12):
        Wc = [h + (t - h).abs() * torch.randn(h.shape, device=DEV, dtype=dt, generator=g) for h, t in zip(H, Tw)]
        acc.append(own_acc(Wc))
    print(f"   (1) Hilton's metric, gap as noise (12 copies): predicted {100*np.mean(acc):.2f}% (copies {100*min(acc):.1f}-{100*max(acc):.1f}) against the trained {100*a_T:.2f}%")
    for s in (0.25, 0.5, 0.75):
        print(f"       straight path H + {s} (T - H): {100*own_acc([h + s * (t - h) for h, t in zip(H, Tw)]):.2f}%", flush=True)
    for name, idx in G.items():
        idx_t = torch.tensor(idx, device=DEV); rest = torch.tensor([j for j in range(16) if j not in idx], device=DEV)
        def patch(cols):
            def step(hT, hH, x):
                hH2 = torch.relu(hH @ H[1].T + x * H[0]); hT2 = torch.relu(hT @ Tw[1].T + x * Tw[0])
                hT2[:, cols] = hH2[:, cols]; return hT2, hH2
            return step
        print(f"   (2) {name:28s} hand-built values patched in: {100*grid_acc(patch(idx_t)):6.2f}%   everything else patched in: {100*grid_acc(patch(rest)):6.2f}%", flush=True)

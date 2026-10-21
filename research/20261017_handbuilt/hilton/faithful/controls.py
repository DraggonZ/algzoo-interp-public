"""Controls (grid 40 x 48 x 48, no sampling):
 (a) how often the first twin (gains 10, matched) gives the trained network's answer;
 (b) the trained network itself plus noise of the same size as the twin's gap (each weight + N(0, |T - H|^2)), 8 copies:
     does noise that large destroy any network?;
 (c) for the best store rule (age table + straight-line residual): share of lists it fixes (trained wrong, rule right),
     breaks (trained right, rule wrong) and changes between two wrong answers."""
import sys
import numpy as np, torch
sys.path.insert(0, "../propagate"); sys.path.insert(0, "../topdown")
import full_mean, adf
DEV = "cuda"; dt = torch.float64
T = adf.net(None)
V, wt = full_mean.grid(40, 48); V = torch.tensor(V, device=DEV, dtype=dt); wt = torch.tensor(wt, device=DEV, dtype=dt)


def batches():
    nV = len(wt)
    for i in range(0, 720, 16):
        tt = full_mean.T[i:i + 16]; pos = tt.repeat_interleave(nV, 0); val = V.repeat(len(tt), 1); n = len(pos)
        bmu, _ = adf.background(val[:, 2]); X = bmu[:, None].repeat(1, 10); ar = torch.arange(n, device=DEV)
        for k in range(3): X[ar, pos[:, k]] = val[:, k]
        yield X, pos[:, 1], wt.repeat(len(tt))


def answers(W, X):
    h = torch.zeros(len(X), 16, device=DEV, dtype=dt)
    for t in range(10): h = torch.relu(h @ W[1].T + X[:, t:t + 1] * W[0])
    return (h @ W[2].T).argmax(1)


Z = wt.sum().item() * 720
d = np.load("../../bridge/v18ts_matched2.npz"); H = [torch.tensor(d[k], device=DEV, dtype=dt) for k in ("wi", "wh", "wo")]
agr = sum(((answers(H, X) == answers(T, X)).to(dt) * w).sum().item() for X, y, w in batches()) / Z
print(f"(a) first twin (gains 10) gives the trained network's answer on {100*agr:.2f}% of lists")
g = torch.Generator(device=DEV).manual_seed(9); accs = []
for c in range(8):
    Wc = [t + (t - h).abs() * torch.randn(t.shape, device=DEV, dtype=dt, generator=g) for t, h in zip(T, H)]
    accs.append(sum(((answers(Wc, X) == y).to(dt) * w).sum().item() for X, y, w in batches()) / Z)
print(f"(b) trained network + noise the size of the twin's gap: accuracy {100*np.mean(accs):.2f}% (copies {100*min(accs):.1f}-{100*max(accs):.1f})")
import store_ladder2 as s2, store_ladder as sl
fix = brk = oth = 0.0
for X, y, w in batches():
    a0 = answers(T, X); a1 = s2.run4(X).argmax(1)
    fix += (((a0 != y) & (a1 == y)).to(dt) * w).sum().item(); brk += (((a0 == y) & (a1 != y)).to(dt) * w).sum().item()
    oth += (((a0 != y) & (a1 != y) & (a1 != a0)).to(dt) * w).sum().item()
print(f"(c) best store rule: fixes {100*fix/Z:.2f}% of lists, breaks {100*brk/Z:.2f}%, changes one wrong answer to another {100*oth/Z:.2f}%")

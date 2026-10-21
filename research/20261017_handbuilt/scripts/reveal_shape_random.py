"""Shape of votes on random lists, from the exact local slopes: J[k, t] = d score_k / d x_t inside each list's
on/off pattern (centred over positions 0-8). Averaged by offset k - t for: old largests (beaten records), kept
non-records, and (for contrast) the current largest. Lists split by fillers: random, versus all other numbers -0.5."""
import numpy as np, torch as th
from vote_ledger import ledger

dev = "cuda"
th.manual_seed(5)
X = th.randn(400_000, 10, device=dev, dtype=th.float64)
_, V = ledger(X)
J = (V / X[:, :, None]).cpu().numpy()          # (n, t, k), centred over all 10 positions
Xn = X.cpu().numpy(); n = 10
z = np.zeros((len(Xn), 2))
m1p = np.stack([np.max(np.c_[z, Xn[:, :t]], 1) for t in range(n)], 1)
m2p = np.stack([np.sort(np.c_[z, Xn[:, :t]], 1)[:, -2] for t in range(n)], 1)
record = Xn > m1p; final = Xn == Xn.max(1, keepdims=True)
kept = ~record & (Xn >= 0.75 * m2p) & (Xn > 0)
np.set_printoptions(precision=1, suppress=True, linewidth=150)
for name, mask in (("old largest", record & ~final), ("kept number", kept), ("current largest", final)):
    prof = np.zeros(17); cnt = np.zeros(17)
    for t in range(1, 8):
        rows = mask[:, t]
        Jt = J[rows, t, :9]; Jt = Jt - Jt.mean(1, keepdims=True)
        for k in range(9):
            prof[k - t + 8] += Jt[:, k].sum(); cnt[k - t + 8] += rows.sum()
    prof /= np.maximum(cnt, 1)
    print(f"{name:16s} slope by offset -4..+4: {prof[4:13]}")

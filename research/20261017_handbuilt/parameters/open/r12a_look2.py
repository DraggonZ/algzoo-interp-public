"""r12a look 2: the readout push through the ReLU, by regression on states (no answers, no ARC readout row k).
Fair votes: row k reads the final state as row k-1 reads it one step later. Instead of linearising with a fixed
on-pattern and a gain, solve r_k = argmin E[(r . h_10 - row_{k-1} . h_11)^2] with h_11 = ReLU(W h_10 + w x), x a
fresh standard normal. Compare r_k with ARC's row k against the current linear push (gain 0.925)."""
import numpy as np
import torch as th
import r12a_x as X

WI, WH, WO = X.WI, X.WH, X.WO
N = 400_000
g = th.Generator().manual_seed(101)
x = th.randn(N, 11, generator=g, dtype=th.float64)
Wi, Wh = th.tensor(WI), th.tensor(WH)
h = th.zeros(N, 16, dtype=th.float64)
for t in range(10):
    h = th.relu(h @ Wh.T + x[:, t, None] * Wi)
h10 = h.numpy()
h11 = th.relu(h @ Wh.T + x[:, 10, None] * Wi).numpy()
h11z = np.maximum(h10 @ WH.T, 0)   # no fresh number
SRC = [i for i in range(16) if i not in (4, 5, 9)]
TGT = [0, 3, 10, 11, 12, 13, 14, 15, 1, 6, 7, 8, 2, 5]
Bm = np.zeros((16, 16)); Bm[SRC] = WH[SRC]
print("on-fraction at h10:", np.round((h10 > 0).mean(0), 2))
G = h10.T @ h10 / N
for name, H1 in (("fresh x", h11), ("x=0", h11z)):
    print(name)
    for k in range(1, 10):
        y = H1 @ WO[k - 1]
        r = np.linalg.solve(G + 1e-9 * np.eye(16), h10.T @ y / N)
        lin = 0.925 * WO[k - 1] @ Bm
        e_r = np.linalg.norm(r[TGT] - WO[k, TGT]) / np.linalg.norm(WO[k, TGT])
        e_l = np.linalg.norm(lin[TGT] - WO[k, TGT]) / np.linalg.norm(WO[k, TGT])
        e_r16 = np.linalg.norm(r - WO[k]) / np.linalg.norm(WO[k])
        # best gain on regression
        gb = (r[TGT] @ WO[k, TGT]) / (r[TGT] @ r[TGT])
        print(f"  row {k}: regression err (push cols) {e_r:.3f}, all 16 {e_r16:.3f}, best scale {gb:.3f}; linear push {e_l:.3f}")

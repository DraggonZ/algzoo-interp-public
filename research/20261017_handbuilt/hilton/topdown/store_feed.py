"""What does the rest of the network feed the store (2)? u(t) = sum over neurons j other than 2 and 4 of W[2, j] h_j(t-1).
By steps since the largest arrived: u / largest (median, 10-90%), and the split into the cancelling neurons (1, 6, 7, 8),
the memory (0, 3, 10-15) and the gates (5, 9). Also: the store's value relative to 17.6 x largest with and without u."""
import numpy as np
from algzoo.core import example
wi, wh, wo = (np.asarray(p, float) for p in example())
rng = np.random.default_rng(8); N = 300000; X = rng.standard_normal((N, 10))
h = np.zeros((N, 16)); H = [h]
for t in range(10): h = np.maximum(h @ wh.T + X[:, t:t + 1] * wi, 0); H.append(h)
H = np.stack(H, 1)                                     # H[:, t] = state after t steps
M = np.maximum.accumulate(np.maximum(X, 0), 1); arg = np.zeros((N, 10), int); cur = np.full(N, -1)
for t in range(10): cur = np.where(X[:, t] > (M[:, t - 1] if t else 0), t, cur); arg[:, t] = cur
grp = {"cancelling": [1, 6, 7, 8], "memory": [0, 3, 10, 11, 12, 13, 14, 15], "gates": [5, 9]}
print("age | u/largest median (10-90%) | cancelling, memory, gates (medians) | store/largest")
for A in range(1, 9):
    rows = []
    for t in range(1, 10):
        prev = M[:, t - 1]; age = t - arg[:, t - 1]; ok = (arg[:, t - 1] >= 0) & (prev > 0.3) & (age == A)
        if ok.sum() == 0: continue
        hp = H[ok, t]                                  # state after t steps = input to step t+1 ... use state feeding step t:
        hin = H[ok, t - 1 + 1 - 1]
        parts = {g: hin[:, idx] @ wh[2, idx] / prev[ok] for g, idx in grp.items()}
        rows.append((sum(parts.values()), parts, H[ok, t, 2] / prev[ok]))
    u = np.concatenate([r[0] for r in rows]); P = {g: np.concatenate([r[1][g] for r in rows]) for g in grp}; s = np.concatenate([r[2] for r in rows])
    print(f"{A} | {np.median(u):+.2f} ({np.quantile(u,.1):+.2f} to {np.quantile(u,.9):+.2f}) | " + ", ".join(f"{np.median(P[g]):+.2f}" for g in grp) + f" | {np.median(s):.2f}")

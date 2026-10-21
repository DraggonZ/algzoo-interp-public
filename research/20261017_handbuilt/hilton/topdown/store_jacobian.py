"""Is the store's value a fixed weighted sum of the list's numbers in each situation? The network has no biases and is
piecewise linear, so on each list exactly S_t = sum_s J(t, s) x_s, where J(t, s) is the product of the weights along
that list's own on/off pattern. Per age of the largest (steps since it arrived), group the coefficients by the role of
each number: the largest (L), the previous largest (P), the numbers since L arrived (by how many steps after L), and
all other earlier numbers. Report the median and spread of each coefficient, and how much of the store's value a
table of medians (coefficient per age and role) explains."""
import numpy as np, torch
from algzoo.core import example
DEV = "cuda"; dt = torch.float64
wi, wh, wo = (torch.tensor(np.asarray(p, float), device=DEV, dtype=dt) for p in example())
N = 200000; X = torch.randn(N, 10, device=DEV, dtype=dt, generator=torch.Generator(device=DEV).manual_seed(41))
h = torch.zeros(N, 16, device=DEV, dtype=dt); J = torch.zeros(N, 10, 16, device=DEV, dtype=dt); S, JS = [], []
for t in range(10):
    a = h @ wh.T + X[:, t:t + 1] * wi; on = (a > 0).to(dt)
    J = J @ wh.T; J[:, t, :] += wi; J = J * on[:, None, :]; h = a * on
    S.append(h[:, 2].clone()); JS.append(J[:, :, 2].clone())
S = torch.stack(S, 1).cpu().numpy(); JS = torch.stack(JS, 1).cpu().numpy()      # JS[n, t, s]: d store(t) / d x_s
Xn = X.cpu().numpy()
print("check: store = sum_s J x_s exactly:", np.abs((JS * Xn[:, None, :]).sum(2) - S).max())
# roles, per step t: largest before step t (position r), previous largest (position rp)
r = np.full(N, -1); rp = np.full(N, -1); m = np.zeros(N); R = np.zeros((N, 10), int); RP = np.zeros((N, 10), int)
for t in range(10):
    R[:, t] = r; RP[:, t] = rp; new = Xn[:, t] > m; rp = np.where(new, r, rp); m = np.where(new, Xn[:, t], m); r = np.where(new, t, r)
ar = np.arange(N)
print("age | coefficient on the largest: median (10-90%) | on the previous largest | on the k-th number since (k = 1, 2, ...) | on other earlier numbers (median of |coef|) | share of store explained by the median table")
for a in range(1, 9):
    rows = []
    for t in range(a, 10):
        ok = (R[:, t] == t - a) & (R[:, t] >= 0) & (Xn[ar, np.maximum(R[:, t], 0)] > 0.3)
        idx = np.where(ok)[0]
        if len(idx) == 0: continue
        rows.append((t, idx))
    cL = np.concatenate([JS[idx, t, t - a] for t, idx in rows])
    hasP = np.concatenate([RP[idx, t] >= 0 for t, idx in rows]); cP = np.concatenate([JS[idx, t, np.maximum(RP[idx, t], 0)] for t, idx in rows])[hasP]
    since = [np.concatenate([JS[idx, t, t - a + k] for t, idx in rows]) for k in range(1, a)]
    others = np.concatenate([np.abs(np.where((np.arange(10)[None] < t - a) & (np.arange(10)[None] != RP[idx, t][:, None]), JS[idx, t, :], 0)).max(1) for t, idx in rows])
    # median table prediction
    med = {"L": np.median(cL), "P": np.median(cP) if len(cP) else 0.0, "since": [np.median(s) for s in since]}
    pred, true = [], []
    for t, idx in rows:
        p = med["L"] * Xn[idx, t - a] + np.where(RP[idx, t] >= 0, med["P"] * Xn[idx, np.maximum(RP[idx, t], 0)], 0)
        for k in range(1, a): p = p + med["since"][k - 1] * Xn[idx, t - a + k]
        pred.append(p); true.append(S[idx, t])
    pred, true = np.concatenate(pred), np.concatenate(true)
    r2 = 1 - ((true - pred) ** 2).mean() / true.var()
    q = lambda v: f"{np.median(v):+6.2f} ({np.quantile(v, .1):+6.2f} to {np.quantile(v, .9):+6.2f})"
    print(f"{a} | {q(cL)} | {q(cP) if len(cP) else '   -'} | " + ", ".join(f"{np.median(s):+.2f}" for s in since) + f" | {np.median(others):.2f} | {r2:.3f}", flush=True)

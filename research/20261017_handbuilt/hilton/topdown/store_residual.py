"""What does the store's remaining residual (store - r(age) L) track? Per age, the share of its variance explained by
simple list quantities (each alone, then together, least squares with a constant): P = the previous largest (before L),
m2 = the second-largest so far, the numbers since L arrived (each, by position since), and the largest number since L."""
import numpy as np, torch
import store_ladder as sl
X = torch.randn(300000, 10, device="cuda", dtype=torch.float64, generator=torch.Generator(device="cuda").manual_seed(21))
wi, wh, wo = sl.W; h = torch.zeros(len(X), 16, device="cuda", dtype=torch.float64); S = []
for t in range(10): h = torch.relu(h @ wh.T + X[:, t:t + 1] * wi); S.append(h[:, 2])
S = torch.stack(S, 1).cpu().numpy(); Xn = X.cpu().numpy(); L, A = (v.cpu().numpy() for v in sl.track(X)); r = sl.r.cpu().numpy()
n = len(Xn); P = np.zeros((n, 10)); M2 = np.zeros((n, 10))
m = np.zeros(n); p = np.zeros(n); m2 = np.zeros(n)
for t in range(10):
    P[:, t] = p; M2[:, t] = m2; x = Xn[:, t]; new = x > m
    m2 = np.where(new, m, np.maximum(m2, x)); p = np.where(new, m, p); m = np.where(new, x, m)
def r2(F, y):
    F = np.c_[F, np.ones(len(y))]; c, *_ = np.linalg.lstsq(F, y, rcond=None); return 1 - ((y - F @ c) ** 2).mean() / y.var(), c
for a in range(2, 9):
    rows = [(t, (A[:, t] == a) & (L[:, t] > 0.3)) for t in range(a, 10)]
    y = np.concatenate([S[ok, t] - r[a] * L[ok, t] for t, ok in rows]) 
    Lr = np.concatenate([L[ok, t] for t, ok in rows]); Pp = np.concatenate([P[ok, t] for t, ok in rows]); Mm = np.concatenate([M2[ok, t] for t, ok in rows])
    since = np.concatenate([np.stack([Xn[ok, t - a + 1 + j] for j in range(a - 1)], 1) if a > 1 else np.zeros((ok.sum(), 0)) for t, ok in rows])
    tmax = since.max(1) if since.shape[1] else np.zeros(len(y))
    out = {"P": r2(Pp[:, None], y)[0], "m2": r2(Mm[:, None], y)[0], "numbers since": r2(np.c_[since, np.maximum(since, 0)], y)[0],
           "L, P, m2, numbers since": r2(np.c_[Lr, Pp, Mm, since, np.maximum(since, 0)], y)[0]}
    print(f"age {a}: residual SD {y.std():.2f}; share explained: " + ", ".join(f"{k} {v:.2f}" for k, v in out.items()), flush=True)

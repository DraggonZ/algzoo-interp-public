"""Store ladder, next rungs. rule 4: r(age) L + a straight-line residual formula per age in L, P (previous largest),
m2 (second-largest so far) and the numbers since L arrived (each and its positive part), fitted on 300,000 random lists
(seed 21; ages 2-8; ages 1 and 9 keep the table only). rule 5: rule 4 + its remaining residual as random noise of the
measured size per age. Grid as in store_ladder.py."""
import numpy as np, torch
import store_ladder as sl
DEV = "cuda"; dt = torch.float64
wi, wh, wo = sl.W


def feats(X):
    """per step t: L, A, P, M2 [n,10] (before step t)."""
    n = X.shape[0]; L, A = sl.track(X)
    P = torch.zeros(n, 10, device=DEV, dtype=dt); M2 = torch.zeros(n, 10, device=DEV, dtype=dt)
    m = torch.zeros(n, device=DEV, dtype=dt); p = torch.zeros(n, device=DEV, dtype=dt); m2 = torch.zeros(n, device=DEV, dtype=dt)
    for t in range(10):
        P[:, t] = p; M2[:, t] = m2; x = X[:, t]; new = x > m
        m2 = torch.where(new, m, torch.maximum(m2, x)); p = torch.where(new, m, p); m = torch.where(new, x, m)
    return L, A, P, M2


def design(X, L, P, M2, t, a, idx):
    since = [X[idx, t - a + 1 + j] for j in range(a - 1)]
    cols = [L[idx, t], P[idx, t], M2[idx, t]] + since + [s.clamp(min=0) for s in since] + [torch.ones(len(idx), device=DEV, dtype=dt)]
    return torch.stack(cols, 1)


# fit
X = torch.randn(300000, 10, device=DEV, dtype=dt, generator=torch.Generator(device=DEV).manual_seed(21))
h = torch.zeros(len(X), 16, device=DEV, dtype=dt); S = []
for t in range(10): h = torch.relu(h @ wh.T + X[:, t:t + 1] * wi); S.append(h[:, 2])
S = torch.stack(S, 1); L, A, P, M2 = feats(X)
coef, sd = {}, torch.zeros(10, device=DEV, dtype=dt)
for a in range(2, 9):
    Fs, ys = [], []
    for t in range(a, 10):
        idx = torch.where((A[:, t] == a) & (L[:, t] > 0.3))[0]
        Fs.append(design(X, L, P, M2, t, a, idx)); ys.append(S[idx, t] - sl.r[a] * L[idx, t])
    F, y = torch.cat(Fs), torch.cat(ys); c = torch.linalg.lstsq(F, y[:, None]).solution[:, 0]; coef[a] = c
    res = (y - F @ c) / torch.cat([L[torch.where((A[:, t] == a) & (L[:, t] > 0.3))[0], t] for t in range(a, 10)])
    sd[a] = (res.quantile(0.84) - res.quantile(0.16)) / 2
print("coefficients per age:", sum(len(c) for c in coef.values()), "| remaining spread (units of L):", " ".join(f"{a}:{sd[a].item():.2f}" for a in range(2, 9)))


def run4(Xg, noise=False, gen=None):
    n = Xg.shape[0]; h = torch.zeros(n, 16, device=DEV, dtype=dt); L, A, P, M2 = feats(Xg)
    for t in range(10):
        h = torch.relu(h @ wh.T + Xg[:, t:t + 1] * wi)
        v = sl.r[A[:, t].clamp(max=9)] * L[:, t]
        for a in range(2, min(9, t + 1)):
            idx = torch.where(A[:, t] == a)[0]
            if len(idx): v[idx] += design(Xg, L, P, M2, t, a, idx) @ coef[a]
        if noise:
            v = v + L[:, t] * sd[A[:, t].clamp(max=9)] * torch.randn(n, device=DEV, dtype=dt, generator=gen)
        h[:, 2] = torch.relu(v)
    return h @ wo.T


def grid_acc(noise=False):
    nV = len(sl.wtt); tot = 0.0; gen = torch.Generator(device=DEV).manual_seed(3); reps = 8 if noise else 1
    for i in range(0, 720, 16):
        tt = sl.full_mean.T[i:i + 16]; pos = tt.repeat_interleave(nV, 0); val = sl.Vt.repeat(len(tt), 1); n = len(pos)
        bmu = -torch.exp(-val[:, 2] ** 2 / 2) / np.sqrt(2 * np.pi) / (0.5 * (1 + torch.erf(val[:, 2] / np.sqrt(2))))
        Xg = bmu[:, None].repeat(1, 10); ar = torch.arange(n, device=DEV)
        for k in range(3): Xg[ar, pos[:, k]] = val[:, k]
        for _ in range(reps):
            tot += ((run4(Xg, noise, gen).argmax(1) == pos[:, 1]).to(dt) * sl.wtt.repeat(len(tt))).sum().item() / reps
    return tot / (sl.wtt.sum().item() * 720)


if __name__ == "__main__":
    for name, nz in [("rule 4: age table + straight-line residual formula", False), ("rule 5: rule 4 + remaining residual as random noise", True)]:
        acc = grid_acc(nz); print(f"{name:52s} {100*acc:6.2f}%  ({100*(acc-0.9523):+.2f})", flush=True)

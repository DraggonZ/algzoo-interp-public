"""Residual explanations of the store (neuron 2), scored without sampling. At every step the store's value is replaced
by a rule; all other neurons run on their own weights and see the replaced value. Grid: ../propagate/full_mean.py
40 x 48 x 48 (intact network 95.23% on it). L = largest number before this step (the store lags one step),
a = steps since L arrived (1 = it arrived at the previous step).
  rule 0: 17.6 L                       (running maximum, no leak)
  rule 1: 17.6 L x 0.969^(a-1)         (main story: the store's own 3% leak per step)
  rule 2: r(a) L, r = the median ratio store / L at each age on random lists (9 numbers; the first residual: the
          memory's and cancelling neurons' pushes, as an age table)
  rule 3: rule 2 + noise: the remaining residual (store - r(a) L) / L drawn at random from its measured spread at that
          age (normal, SD from random lists), 8 draws per grid list; mean accuracy.
  rule 4: rule 2 + the true remaining residual for the memory and cancelling neurons' pushes only along ... (not yet)
The ratio table and spreads come from 300,000 random lists (seed 8)."""
import sys
import numpy as np, torch
sys.path.insert(0, "../propagate")
import full_mean
from algzoo.core import example
DEV = "cuda"; dt = torch.float64
W = [torch.tensor(np.asarray(p, float), device=DEV, dtype=dt) for p in example()]
wi, wh, wo = W


def track(X):
    """L (largest before each step, floor 0) and a (steps since it arrived; 0 if none yet), per step, for lists X [n,10]."""
    n = X.shape[0]; L = torch.zeros(n, 10, device=DEV, dtype=dt); A = torch.zeros(n, 10, device=DEV, dtype=torch.long)
    m = torch.zeros(n, device=DEV, dtype=dt); r = torch.full((n,), -1, device=DEV, dtype=torch.long)
    for t in range(10):
        L[:, t] = m; A[:, t] = torch.where(r >= 0, t - r, 0)
        new = X[:, t] > m; m = torch.where(new, X[:, t], m); r = torch.where(new, t, r)
    return L, A


def run(X, rule=None, noise=None, gen=None):
    n = X.shape[0]; h = torch.zeros(n, 16, device=DEV, dtype=dt)
    L, A = track(X)
    for t in range(10):
        h = torch.relu(h @ wh.T + X[:, t:t + 1] * wi)
        if rule is not None:
            v = rule(L[:, t], A[:, t])
            if noise is not None:
                v = v + L[:, t] * noise[A[:, t].clamp(max=9)] * torch.randn(n, device=DEV, dtype=dt, generator=gen)
            h[:, 2] = torch.relu(v)
    return h @ wo.T


# ratio table and spread from random lists
X = torch.randn(300000, 10, device=DEV, dtype=dt, generator=torch.Generator(device=DEV).manual_seed(8))
h = torch.zeros(len(X), 16, device=DEV, dtype=dt); S = []
for t in range(10): h = torch.relu(h @ wh.T + X[:, t:t + 1] * wi); S.append(h[:, 2])
S = torch.stack(S, 1); L, A = track(X)
r = torch.zeros(10, device=DEV, dtype=dt); sd = torch.zeros(10, device=DEV, dtype=dt)
for a in range(1, 10):
    ok = (A == a) & (L > 0.3)
    q = S[ok] / L[ok]; r[a] = q.median(); res = q - r[a]
    sd[a] = (res.quantile(0.84) - res.quantile(0.16)) / 2
print("ratio by age:", " ".join(f"{a}:{r[a].item():.2f}" for a in range(1, 10)))
print("residual spread (robust SD, units of L):", " ".join(f"{a}:{sd[a].item():.2f}" for a in range(1, 10)))

V, wt = full_mean.grid(40, 48); Vt = torch.tensor(V, device=DEV, dtype=dt); wtt = torch.tensor(wt, device=DEV, dtype=dt)


def grid_acc(**kw):
    nV = len(wtt); tot = 0.0; gen = torch.Generator(device=DEV).manual_seed(3)
    reps = 8 if kw.get("noise") is not None else 1
    for i in range(0, 720, 16):
        tt = full_mean.T[i:i + 16]; pos = tt.repeat_interleave(nV, 0); val = Vt.repeat(len(tt), 1); n = len(pos)
        bmu = -torch.exp(-val[:, 2] ** 2 / 2) / np.sqrt(2 * np.pi) / (0.5 * (1 + torch.erf(val[:, 2] / np.sqrt(2))))
        Xg = bmu[:, None].repeat(1, 10); ar = torch.arange(n, device=DEV)
        for k in range(3): Xg[ar, pos[:, k]] = val[:, k]
        for _ in range(reps):
            tot += ((run(Xg, gen=gen, **kw).argmax(1) == pos[:, 1]).to(dt) * wtt.repeat(len(tt))).sum().item() / reps
    return tot / (wtt.sum().item() * 720)


if __name__ == "__main__":
    base = grid_acc()
    print(f"intact network: {100*base:.2f}%")
    rules = [("rule 0: 17.6 x largest, no leak", lambda L, a: 17.6 * L),
             ("rule 1: 17.6 x largest x 0.969^(age-1) (own leak)", lambda L, a: 17.6 * L * 0.969 ** (a - 1).clamp(min=0).to(dt)),
             ("rule 2: age table r(age) x largest", lambda L, a: r[a.clamp(max=9)] * L)]
    for name, f in rules:
        acc = grid_acc(rule=f); print(f"{name:52s} {100*acc:6.2f}%  ({100*(acc-base):+.2f})", flush=True)
    acc = grid_acc(rule=rules[2][1], noise=sd); print(f"{'rule 3: age table + residual as random noise':52s} {100*acc:6.2f}%  ({100*(acc-base):+.2f})")

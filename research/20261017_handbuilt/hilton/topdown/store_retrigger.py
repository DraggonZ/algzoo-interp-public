"""A short mechanistic rule for the store: it holds the most recent number that beat a cutoff falling with the stored
value's age. Tracked quantity V (the stored number) and its age b: at each step, if x > theta(b) V then V <- x, b <- 0.
Store = c(b) V + q(b) P, P = the previous stored number. theta from the detector's own weights: the detector fires when
10.157 x > 0.492 S + 0.986 D', i.e. (D' = 0) x > 0.0484 S; with S = c(b) V that is theta(b) = 0.0484 c(b).
c(b), q(b): medians of the store's exact per-number weights by age (store_jacobian.py's method, now by the age of V).
Rules scored on the grid (40 x 48 x 48, no sampling) by mistakes reproduced, agreement and accuracy; store replaced alone.
  A: largest-so-far rule (V = the largest, no re-trigger), store = c(age) L            (the age table, 9 numbers)
  B: re-trigger rule, store = c(b) V                                                    (9 numbers, theta derived)
  C: re-trigger rule, store = c(b) V + q(b) P                                           (18 numbers)"""
import numpy as np, torch
import store_ladder as sl
DEV = "cuda"; dt = torch.float64
wi, wh, wo = sl.W


def track_V(X, theta):
    n = X.shape[0]; V = torch.zeros(n, 10, device=DEV, dtype=dt); B = torch.zeros(n, 10, device=DEV, dtype=torch.long); P = torch.zeros(n, 10, device=DEV, dtype=dt)
    v = torch.zeros(n, device=DEV, dtype=dt); b = torch.full((n,), -1, device=DEV, dtype=torch.long); p = torch.zeros(n, device=DEV, dtype=dt)
    for t in range(10):
        V[:, t] = v; B[:, t] = torch.where(b >= 0, b + 1, 0); P[:, t] = p      # values going into step t (store lags one step)
        th = theta[(b + 1).clamp(min=0, max=9)]
        new = (X[:, t] > th * v) & (X[:, t] > 0)
        p = torch.where(new, v, p); v = torch.where(new, X[:, t], v); b = torch.where(new, 0, b + 1)
    return V, B, P


# exact per-number weights of the store, grouped by the age of V (iterate: theta from c, c from data with that theta)
N = 200000; X = torch.randn(N, 10, device=DEV, dtype=dt, generator=torch.Generator(device=DEV).manual_seed(41))
h = torch.zeros(N, 16, device=DEV, dtype=dt); J = torch.zeros(N, 10, 16, device=DEV, dtype=dt); S, JS = [], []
for t in range(10):
    a = h @ wh.T + X[:, t:t + 1] * wi; on = (a > 0).to(dt); J = J @ wh.T; J[:, t, :] += wi; J = J * on[:, None, :]; h = a * on
    S.append(h[:, 2].clone()); JS.append(J[:, :, 2].clone())
S = torch.stack(S, 1); JS = torch.stack(JS, 1)
theta = torch.full((10,), 0.853, device=DEV, dtype=dt); theta[0] = 1.0
for it in range(3):
    V, B, P = track_V(X, theta)
    # position of V and P
    c = torch.zeros(10, device=DEV, dtype=dt); q = torch.zeros(10, device=DEV, dtype=dt)
    for bb in range(1, 10):
        ok = (B == bb) & (V > 0.3)
        if ok.sum() < 100: continue
        ratio = S[ok] / V[ok]; c[bb] = ratio.median()
        res = S[ok] - c[bb] * V[ok]; Pk = P[ok]; m = Pk > 0.1
        q[bb] = (res[m] / Pk[m]).median() if m.sum() > 100 else 0.0
    theta = (0.0484 * c).clamp(max=1.0); theta[0] = 1.0; theta[1:] = torch.where(c[1:] > 0, theta[1:], 0.853)
    print(f"iteration {it}: c(b) = " + " ".join(f"{v:.1f}" for v in c[1:9].tolist()) + " | theta(b) = " + " ".join(f"{v:.2f}" for v in theta[1:9].tolist()) + " | q(b) = " + " ".join(f"{v:.2f}" for v in q[1:9].tolist()), flush=True)
# how much of the store each rule explains on random lists
L, A = sl.track(X)
for name, pred in [("A: c(age) L", sl.r[A.clamp(max=9)] * L), ("B: c(b) V", c[B.clamp(max=9)] * V), ("C: c(b) V + q(b) P", c[B.clamp(max=9)] * V + q[B.clamp(max=9)] * P)]:
    ok = A >= 1
    print(f"{name:22s} share of the store's variance explained on random lists: {1 - ((S - pred)[ok] ** 2).mean().item() / S[ok].var().item():.4f}")


def run(Xg, rule):
    n = Xg.shape[0]; h = torch.zeros(n, 16, device=DEV, dtype=dt); L, A = sl.track(Xg); V, B, P = track_V(Xg, theta)
    for t in range(10):
        h = torch.relu(h @ wh.T + Xg[:, t:t + 1] * wi)
        if rule == "A": v = sl.r[A[:, t].clamp(max=9)] * L[:, t]
        elif rule == "B": v = c[B[:, t].clamp(max=9)] * V[:, t]
        else: v = c[B[:, t].clamp(max=9)] * V[:, t] + q[B[:, t].clamp(max=9)] * P[:, t]
        h[:, 2] = torch.relu(v)
    return h @ wo.T


nV = len(sl.wtt); res = {k: np.zeros(3) for k in "ABC"}; Zw = Z = 0.0
for i in range(0, 720, 16):
    tt = sl.full_mean.T[i:i + 16]; pos = tt.repeat_interleave(nV, 0); val = sl.Vt.repeat(len(tt), 1); n = len(pos)
    bmu = -torch.exp(-val[:, 2] ** 2 / 2) / np.sqrt(2 * np.pi) / (0.5 * (1 + torch.erf(val[:, 2] / np.sqrt(2))))
    Xg = bmu[:, None].repeat(1, 10); ar = torch.arange(n, device=DEV)
    for k in range(3): Xg[ar, pos[:, k]] = val[:, k]
    w = sl.wtt.repeat(len(tt)); y = pos[:, 1]; a0 = sl.run(Xg).argmax(1); wr = a0 != y; Zw += (w * wr).sum().item(); Z += w.sum().item()
    for k in "ABC":
        a1 = run(Xg, k).argmax(1)
        res[k] += [(w * (a1 == a0)).sum().item(), (w * (wr & (a1 == a0))).sum().item(), (w * (a1 == y)).sum().item()]
for k, nm in zip("ABC", ["A: largest so far, age table (9 numbers)", "B: re-trigger rule, c(b) V (9 numbers)", "C: re-trigger rule + previous stored number (18)"]):
    print(f"{nm:50s} mistakes reproduced {100*res[k][1]/Zw:5.1f}%   agreement {100*res[k][0]/Z:6.2f}%   accuracy {100*res[k][2]/Z:6.2f}%")

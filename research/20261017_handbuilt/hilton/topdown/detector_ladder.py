"""Residual explanations of the detector (neuron 4), scored without sampling by agreement with the trained network
(and accuracy). At every step the detector's value is replaced by a rule; everything else runs on its own weights.
x = arriving number, L = largest before this step, age a, S = the store's value going into this step, D' = the
detector's previous value. The detector's own weights: input 10.157, from the store -0.492, from itself -0.986.
  d0: 10.16 (x - L)+                      fires only at a new largest, in proportion to the jump
  d1: 10.16 (x - 0.853 L)+                the same threshold the weights give if the store held exactly 17.6 L
  d2: (10.16 x - 0.492 S - 0.986 D')+     its own three weights, reading the real store and itself
  d3: (10.16 x - 0.492 r(a) L - 0.986 D')+   the store taken from its age table instead
  d4: d2 + a straight-line residual per age in L, P, m2 and the numbers since (fitted on 300,000 random lists)."""
import numpy as np, torch
import store_ladder as sl, store_ladder2 as s2
DEV = "cuda"; dt = torch.float64
wi, wh, wo = sl.W


def run(X, rule=None):
    n = X.shape[0]; h = torch.zeros(n, 16, device=DEV, dtype=dt); L, A, P, M2 = s2.feats(X)
    for t in range(10):
        hp = h; h = torch.relu(hp @ wh.T + X[:, t:t + 1] * wi)
        if rule is not None:
            h[:, 4] = torch.relu(rule(t, X, hp, L, A, P, M2))
    return h @ wo.T


def own(t, X, hp, *_):
    return 10.157 * X[:, t] - 0.492 * hp[:, 2] - 0.986 * hp[:, 4]


# residual fit for d4
Xf = torch.randn(300000, 10, device=DEV, dtype=dt, generator=torch.Generator(device=DEV).manual_seed(22))
h = torch.zeros(len(Xf), 16, device=DEV, dtype=dt); Pre, Own = [], []
for t in range(10):
    hp = h; pre = hp @ wh.T + Xf[:, t:t + 1] * wi; h = torch.relu(pre); Pre.append(pre[:, 4]); Own.append(own(t, Xf, hp))
Pre, Own = torch.stack(Pre, 1), torch.stack(Own, 1); L, A, P, M2 = s2.feats(Xf); coef = {}
for a in range(2, 9):
    Fs, ys = [], []
    for t in range(a, 10):
        idx = torch.where(A[:, t] == a)[0]; Fs.append(s2.design(Xf, L, P, M2, t, a, idx)); ys.append(Pre[idx, t] - Own[idx, t])
    coef[a] = torch.linalg.lstsq(torch.cat(Fs), torch.cat(ys)[:, None]).solution[:, 0]
res_all = (Pre - Own)[A >= 1]; print(f"detector pre-activation minus its own three terms: SD {res_all.std().item():.2f} (pre-activation SD {Pre[A >= 1].std().item():.2f})")


def d4(t, X, hp, L, A, P, M2):
    v = own(t, X, hp)
    for a in range(2, min(9, t + 1)):
        idx = torch.where(A[:, t] == a)[0]
        if len(idx): v[idx] += s2.design(X, L, P, M2, t, a, idx) @ coef[a]
    return v


rules = [("d0: 10.16 (x - largest)+", lambda t, X, hp, L, *_: 10.16 * (X[:, t] - L[:, t])),
         ("d1: 10.16 (x - 0.853 largest)+", lambda t, X, hp, L, *_: 10.16 * (X[:, t] - 0.853 * L[:, t])),
         ("d2: own three weights, real store", own),
         ("d3: own weights, store from its age table", lambda t, X, hp, L, A, *_: 10.157 * X[:, t] - 0.492 * sl.r[A[:, t].clamp(max=9)] * L[:, t] - 0.986 * hp[:, 4]),
         ("d4: d2 + straight-line residual", d4)]
if __name__ == "__main__":
    nV = len(sl.wtt); res = {k: [0.0, 0.0] for k, _ in rules}; base = 0.0
    for i in range(0, 720, 16):
        tt = sl.full_mean.T[i:i + 16]; pos = tt.repeat_interleave(nV, 0); val = sl.Vt.repeat(len(tt), 1); n = len(pos)
        bmu = -torch.exp(-val[:, 2] ** 2 / 2) / np.sqrt(2 * np.pi) / (0.5 * (1 + torch.erf(val[:, 2] / np.sqrt(2))))
        Xg = bmu[:, None].repeat(1, 10); ar = torch.arange(n, device=DEV)
        for k in range(3): Xg[ar, pos[:, k]] = val[:, k]
        w = sl.wtt.repeat(len(tt)); a0 = run(Xg).argmax(1); base += ((a0 == pos[:, 1]).to(dt) * w).sum().item()
        for name, f in rules:
            a1 = run(Xg, f).argmax(1)
            res[name][0] += ((a1 == pos[:, 1]).to(dt) * w).sum().item(); res[name][1] += ((a1 == a0).to(dt) * w).sum().item()
    Z = sl.wtt.sum().item() * 720
    print(f"intact trained network: accuracy {100*base/Z:.2f}%")
    for name, (acc, agr) in res.items():
        print(f"{name:44s} agreement {100*agr/Z:6.2f}%   accuracy {100*acc/Z:6.2f}%")

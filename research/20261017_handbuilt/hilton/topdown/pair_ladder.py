"""Store and detector replaced together (the whole largest-so-far block), scored by agreement with the trained
network and accuracy on the grid. Store rules from store_ladder2.py (age table, + straight-line residual); detector
rules from detector_ladder.py (d0 'fires at a new largest', d4 'own weights + straight-line residual'). The detector
rule reads the replaced store."""
import numpy as np, torch
import store_ladder as sl, store_ladder2 as s2, detector_ladder as dl
DEV = "cuda"; dt = torch.float64
wi, wh, wo = sl.W


def store_val(t, Xg, L, A, P, M2, residual):
    v = sl.r[A[:, t].clamp(max=9)] * L[:, t]
    if residual:
        for a in range(2, min(9, t + 1)):
            idx = torch.where(A[:, t] == a)[0]
            if len(idx): v[idx] += s2.design(Xg, L, P, M2, t, a, idx) @ s2.coef[a]
    return v


def run(Xg, srule, drule):
    n = Xg.shape[0]; h = torch.zeros(n, 16, device=DEV, dtype=dt); L, A, P, M2 = s2.feats(Xg)
    for t in range(10):
        hp = h; h = torch.relu(hp @ wh.T + Xg[:, t:t + 1] * wi)
        h[:, 4] = torch.relu(drule(t, Xg, hp, L, A, P, M2))
        h[:, 2] = torch.relu(store_val(t, Xg, L, A, P, M2, srule == "residual"))
    return h @ wo.T


if __name__ == "__main__":
    combos = [("store age table + detector fires at new largest", "table", dl.rules[0][1]),
              ("store with residual + detector fires at new largest", "residual", dl.rules[0][1]),
              ("store with residual + detector with residual", "residual", dl.d4)]
    nV = len(sl.wtt); res = {k: [0.0, 0.0] for k, *_ in combos}
    for i in range(0, 720, 16):
        tt = sl.full_mean.T[i:i + 16]; pos = tt.repeat_interleave(nV, 0); val = sl.Vt.repeat(len(tt), 1); n = len(pos)
        bmu = -torch.exp(-val[:, 2] ** 2 / 2) / np.sqrt(2 * np.pi) / (0.5 * (1 + torch.erf(val[:, 2] / np.sqrt(2))))
        Xg = bmu[:, None].repeat(1, 10); ar = torch.arange(n, device=DEV)
        for k in range(3): Xg[ar, pos[:, k]] = val[:, k]
        w = sl.wtt.repeat(len(tt)); a0 = sl.run(Xg).argmax(1)
        for name, sr, dr in combos:
            a1 = run(Xg, sr, dr).argmax(1); res[name][0] += ((a1 == pos[:, 1]).to(dt) * w).sum().item(); res[name][1] += ((a1 == a0).to(dt) * w).sum().item()
    Z = sl.wtt.sum().item() * 720
    for name, (acc, agr) in res.items(): print(f"{name:54s} agreement {100*agr/Z:6.2f}%   accuracy {100*acc/Z:6.2f}%")

"""Store ladder scored by faithfulness to the trained network, not only by task accuracy: on the grid (no sampling),
the share of lists where the network with the replaced store gives the same answer as the intact trained network
(agreement), alongside accuracy. Rules as in store_ladder.py (0-3) and store_ladder2.py (4-5)."""
import numpy as np, torch
import store_ladder as sl
import store_ladder2 as s2
DEV = "cuda"; dt = torch.float64
rules = [("rule 0: 17.6 x largest", lambda Xg, g: sl.run(Xg, rule=lambda L, a: 17.6 * L)),
         ("rule 1: + own leak", lambda Xg, g: sl.run(Xg, rule=lambda L, a: 17.6 * L * 0.969 ** (a - 1).clamp(min=0).to(dt))),
         ("rule 2: age table", lambda Xg, g: sl.run(Xg, rule=lambda L, a: sl.r[a.clamp(max=9)] * L)),
         ("rule 3: age table + leftover as noise", lambda Xg, g: sl.run(Xg, rule=lambda L, a: sl.r[a.clamp(max=9)] * L, noise=sl.sd, gen=g)),
         ("rule 4: + straight-line residual", lambda Xg, g: s2.run4(Xg)),
         ("rule 5: rule 4 + leftover as noise", lambda Xg, g: s2.run4(Xg, True, g))]
nV = len(sl.wtt); res = {k: [0.0, 0.0] for k, _ in rules}; base = 0.0
gen = torch.Generator(device=DEV).manual_seed(3)
for i in range(0, 720, 16):
    tt = sl.full_mean.T[i:i + 16]; pos = tt.repeat_interleave(nV, 0); val = sl.Vt.repeat(len(tt), 1); n = len(pos)
    bmu = -torch.exp(-val[:, 2] ** 2 / 2) / np.sqrt(2 * np.pi) / (0.5 * (1 + torch.erf(val[:, 2] / np.sqrt(2))))
    Xg = bmu[:, None].repeat(1, 10); ar = torch.arange(n, device=DEV)
    for k in range(3): Xg[ar, pos[:, k]] = val[:, k]
    w = sl.wtt.repeat(len(tt)); a0 = sl.run(Xg).argmax(1); base += ((a0 == pos[:, 1]).to(dt) * w).sum().item()
    for name, f in rules:
        reps = 4 if "noise" in name else 1
        for _ in range(reps):
            a1 = f(Xg, gen).argmax(1)
            res[name][0] += ((a1 == pos[:, 1]).to(dt) * w).sum().item() / reps
            res[name][1] += ((a1 == a0).to(dt) * w).sum().item() / reps
Z = sl.wtt.sum().item() * 720
print(f"intact trained network: accuracy {100*base/Z:.2f}%")
for name, (acc, agr) in res.items():
    print(f"{name:42s} accuracy {100*acc/Z:6.2f}%   agreement with the trained network {100*agr/Z:6.2f}%   (disagree on {100-100*agr/Z:5.2f}%)")

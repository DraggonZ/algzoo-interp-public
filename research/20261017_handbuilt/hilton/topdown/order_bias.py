"""On ordinary lists (largest and second-largest positive) where the third is within 5% of the second: how often does
each network name the third, split by arrival order of the three largest (L, S, T) and by whether S or T arrives right
after L. A generic imprecise reader has no order bias. ARC, five other trained networks (95-96%), v18. Grid 40 x 48 x 48."""
import numpy as np, torch
import store_ladder as sl
from algzoo.core import load
from algzoo.handbuilt_v18 import build_v18
from mistakes_other_nets import T, net
DEV = "cuda"; dt = torch.float64
if __name__ == "__main__":
    v18 = net(T(build_v18().params))
    tr = [net(T(load(f"/data/output/research/algzoo/20261016_run3/models/2nd_argmax_16_10_1073741824_{s}.pth"))) for s in (103, 111, 115, 209, 210)]
    nV = len(sl.wtt); st = {}
    def add(k, v): st[k] = st.get(k, 0.0) + v
    for i in range(0, 720, 16):
        tt = sl.full_mean.T[i:i + 16]; pos = tt.repeat_interleave(nV, 0); val = sl.Vt.repeat(len(tt), 1); n = len(pos)
        bmu = -torch.exp(-val[:, 2] ** 2 / 2) / np.sqrt(2 * np.pi) / (0.5 * (1 + torch.erf(val[:, 2] / np.sqrt(2))))
        X = bmu[:, None].repeat(1, 10); ar = torch.arange(n, device=DEV)
        for k in range(3): X[ar, pos[:, k]] = val[:, k]
        w = sl.wtt.repeat(len(tt))
        m = (val[:, 1] > 0) & ((val[:, 1] - val[:, 2]) / val[:, 1] < 0.05)
        if not m.any(): continue
        X, pos, w = X[m], pos[m], w[m]
        third = {"ARC": (sl.run(X).argmax(1) == pos[:, 2]).double(), "v18": (v18(X).argmax(1) == pos[:, 2]).double(),
                 "trained": torch.stack([(f(X).argmax(1) == pos[:, 2]).double() for f in tr]).mean(0)}
        L, S, Tp = pos[:, 0], pos[:, 1], pos[:, 2]
        groups = {"all": torch.ones_like(L, dtype=torch.bool), "T after S": Tp > S, "T before S": Tp < S,
                  "S right after L": S == L + 1, "T right after L": Tp == L + 1, "both after L": (S > L) & (Tp > L),
                  "both before L": (S < L) & (Tp < L), "L between": ((S < L) & (L < Tp)) | ((Tp < L) & (L < S)),
                  "T at step 9": Tp == 9, "S at step 9": S == 9, "T at step 8": Tp == 8, "S at step 8": S == 8}
        for g, gm in groups.items():
            add((g, "p"), (w * gm).sum().item())
            for nm, v in third.items(): add((g, nm), (w * gm * v).sum().item())
    print("near ties on ordinary lists: third within 5% of the second; share naming the third")
    print(f"{'group':18s} {'share':>6s} {'ARC':>6s} {'trained':>8s} {'v18':>6s}")
    for g in ["all", "T after S", "T before S", "S right after L", "T right after L", "both after L", "both before L", "L between", "T at step 9", "S at step 9", "T at step 8", "S at step 8"]:
        p = st[(g, "p")]; print(f"{g:18s} {100*p/st[('all','p')]:5.1f}% " + " ".join(f"{100*st[(g, nm)]/p:6.1f}%" for nm in ("ARC", "trained", "v18")))

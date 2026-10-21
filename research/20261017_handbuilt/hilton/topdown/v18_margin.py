"""v18 with a margin at overtakings that grows with the jump, as measured in the trained network
(../../bridge/margin_source.log, README there: about 56-131 per unit of jump against a vote of 17-39 per unit of
value, so 1.5-8 votes per unit of jump; mu = 3 taken from the middle, not tuned). The keep gate writes, for a record,
(1 - c) (p - mu d) instead of (1 - c) p - eps L: a record's own position shows the previous largest's vote minus mu
votes per unit of jump. Then ARC's net vote sizes per position at a tie (v18_recency.K_NET) on top.
Neuron order in build_v18: store 0, detect 1, keep 2, flip 3, clock 4-6, flag 7, memory 8-15."""
import numpy as np, torch
import store_ladder as sl
from algzoo.core import load
from algzoo.handbuilt_v18 import build_v18
from mistakes_other_nets import T, net
from v18_recency import K_NET
DEV = "cuda"; dt = torch.float64
S, D, G = 0, 1, 2

def build(mu=3.0, k=np.ones(10), c=0.75, a=1000.0):
    wi, wh, wo = [p.copy() for p in build_v18(c=c, a=a).params]
    A0 = 1 - c
    wh[G, D] -= mu * A0 / a        # record: (1 - c)(p - mu d); d = D / a one step late
    wo[9, D] -= mu * A0 / a        # position 9 reads the keep gate's formula directly
    return wi, wh, wo * k[:, None]

if __name__ == "__main__":
    variants = {"v18": build(0.0), "v18 + margin mu 3": build(3.0), "v18 + margin + net sizes 8, 9": build(3.0, np.r_[np.ones(8), K_NET[8:]]),
                "v18 + margin + net sizes": build(3.0, K_NET), "v18 + margin mu 1.5 + net sizes": build(1.5, K_NET), "v18 + margin mu 8 + net sizes": build(8.0, K_NET)}
    models = {nm: [net(T(W))] for nm, W in variants.items()}
    models["trained 95-96% (mean of 5)"] = [net(T(load(f"/data/output/research/algzoo/20261016_run3/models/2nd_argmax_16_10_1073741824_{s}.pth"))) for s in (103, 111, 115, 209, 210)]
    G_ = ["T after S", "T before S", "both before L", "T at step 9", "T at step 8"]
    nV = len(sl.wtt); st = {}
    def add(key, v): st[key] = st.get(key, 0.0) + v
    for i in range(0, 720, 16):
        tt = sl.full_mean.T[i:i + 16]; pos = tt.repeat_interleave(nV, 0); val = sl.Vt.repeat(len(tt), 1); n = len(pos)
        bmu = -torch.exp(-val[:, 2] ** 2 / 2) / np.sqrt(2 * np.pi) / (0.5 * (1 + torch.erf(val[:, 2] / np.sqrt(2))))
        X = bmu[:, None].repeat(1, 10); ar = torch.arange(n, device=DEV)
        for kk in range(3): X[ar, pos[:, kk]] = val[:, kk]
        w = sl.wtt.repeat(len(tt)); y = pos[:, 1]; a0 = sl.run(X).argmax(1); wr = a0 != y; pp = val[:, 1] > 0
        near = pp & ((val[:, 1] - val[:, 2]) / val[:, 1] < 0.05); L, S_, Tp = pos[:, 0], pos[:, 1], pos[:, 2]
        gm = {"T after S": Tp > S_, "T before S": Tp < S_, "both before L": (S_ < L) & (Tp < L), "T at step 9": Tp == 9, "T at step 8": Tp == 8}
        add("w", (w * wr).sum().item()); add("all", w.sum().item())
        for kd in range(3): add(("arc kind", kd), (w * wr * (a0 == pos[:, kd if kd < 2 else 2])).sum().item())
        for g in G_: add((g, "p"), (w * near * gm[g]).sum().item())
        for nm, fs in models.items():
            for f in fs:
                a1 = f(X).argmax(1); c_ = 1 / len(fs)
                add((nm, "acc"), c_ * (w * (a1 == y)).sum().item()); add((nm, "rep"), c_ * (w * wr * (a1 == a0)).sum().item())
                add((nm, "repL"), c_ * (w * wr * (a0 == L) * (a1 == a0)).sum().item()); add((nm, "repT"), c_ * (w * wr * (a0 == Tp) * (a1 == a0)).sum().item())
                add((nm, "wrongL"), c_ * (w * (a1 == L)).sum().item())
                for g in G_: add((nm, g), c_ * (w * near * gm[g] * (a1 == Tp)).sum().item())
    wL, wT = st[("arc kind", 0)], st[("arc kind", 2)]
    print(f"ARC: names the largest on {100*wL/st['all']:.2f}% of lists, the third on {100*wT/st['all']:.2f}%")
    print(f"{'network':32s} {'accuracy':>8s} {'names L':>8s} | ARC mistakes reproduced: {'all':>5s} {'L':>5s} {'T':>5s} | near ties, share naming T: " + ", ".join(G_))
    for nm in models:
        print(f"{nm:32s} {100*st[(nm,'acc')]/st['all']:7.2f}% {100*st[(nm,'wrongL')]/st['all']:7.2f}% | {'':24s} {100*st[(nm,'rep')]/st['w']:4.1f}% {100*st[(nm,'repL')]/wL:4.1f}% {100*st[(nm,'repT')]/wT:4.1f}% | "
              + "  ".join(f"{100*st[(nm, g)]/st[(g, 'p')]:5.1f}%" for g in G_))

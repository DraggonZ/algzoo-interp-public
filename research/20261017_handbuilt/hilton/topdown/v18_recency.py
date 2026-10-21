"""v18 with ARC's measured vote size per position: each readout row p scaled by k_p, which multiplies score p and keeps
the hiding exact (every term in the row scales together). k from ../../votes/vote_relative_fit.log (a decision-level
fit to ARC, kept numbers; position 0 from the record row since a number at step 0 is always a record or negative).
Variants: full profile; only positions 8 and 9 (read directly, outside the memory); a flat control. No tuning to the
mistakes score. Scores: grid accuracy, ARC's mistakes reproduced (all, positive-second lists), near-tie third share
by arrival group, as in order_bias.py."""
import numpy as np, torch
import store_ladder as sl
from algzoo.core import load
from algzoo.handbuilt_v18 import build_v18
from mistakes_other_nets import T, net
DEV = "cuda"; dt = torch.float64
K_FULL = np.array([1.01, 0.96, 0.96, 0.97, 0.98, 1.00, 1.04, 1.09, 1.28, 1.68])
K_NET = K_FULL + np.array([0, 0, -0., -0.01, -0.01, -0.03, -0.05, -0.1, -0.28, -0.64])  # vote per unit at a tie with the runner-up (x = s)
VARIANTS = {"v18 + net sizes at a tie": K_NET, "v18 + net sizes, 8 and 9 only": np.r_[np.ones(8), K_NET[8:]], "v18": np.ones(10), "v18 + ARC vote sizes": K_FULL, "v18 + sizes at 8, 9 only": np.r_[np.ones(8), K_FULL[8:]],
            "v18 + sizes 1-7 only": np.r_[K_FULL[:8], 1, 1]}
if __name__ == "__main__":
    import sys
    if "--net" in sys.argv: VARIANTS = {k: v for k, v in VARIANTS.items() if "net" in k or k == "v18"}
    wi, wh, wo = build_v18().params
    models = {nm: net(T([wi, wh, wo * k[:, None]])) for nm, k in VARIANTS.items()}
    models["trained 95-96% (mean of 5)"] = [net(T(load(f"/data/output/research/algzoo/20261016_run3/models/2nd_argmax_16_10_1073741824_{s}.pth"))) for s in (103, 111, 115, 209, 210)]
    G = ["T after S", "T before S", "both before L", "T at step 9", "T at step 8"]
    nV = len(sl.wtt); st = {}
    def add(k, v): st[k] = st.get(k, 0.0) + v
    for i in range(0, 720, 16):
        tt = sl.full_mean.T[i:i + 16]; pos = tt.repeat_interleave(nV, 0); val = sl.Vt.repeat(len(tt), 1); n = len(pos)
        bmu = -torch.exp(-val[:, 2] ** 2 / 2) / np.sqrt(2 * np.pi) / (0.5 * (1 + torch.erf(val[:, 2] / np.sqrt(2))))
        X = bmu[:, None].repeat(1, 10); ar = torch.arange(n, device=DEV)
        for k in range(3): X[ar, pos[:, k]] = val[:, k]
        w = sl.wtt.repeat(len(tt)); y = pos[:, 1]; a0 = sl.run(X).argmax(1); wr = a0 != y; pp = val[:, 1] > 0
        near = pp & ((val[:, 1] - val[:, 2]) / val[:, 1] < 0.05); L, S, Tp = pos[:, 0], pos[:, 1], pos[:, 2]
        gm = {"T after S": Tp > S, "T before S": Tp < S, "both before L": (S < L) & (Tp < L), "T at step 9": Tp == 9, "T at step 8": Tp == 8}
        add("w", (w * wr).sum().item()); add("wp", (w * wr * pp).sum().item()); add("all", w.sum().item())
        for g in G: add((g, "p"), (w * near * gm[g]).sum().item())
        for nm, f in models.items():
            outs = [g_(X).argmax(1) for g_ in f] if isinstance(f, list) else [f(X).argmax(1)]
            for a1 in outs:
                c = 1 / len(outs)
                add((nm, "acc"), c * (w * (a1 == y)).sum().item()); add((nm, "rep"), c * (w * wr * (a1 == a0)).sum().item())
                add((nm, "repp"), c * (w * wr * pp * (a1 == a0)).sum().item())
                for g in G: add((nm, g), c * (w * near * gm[g] * (a1 == Tp)).sum().item())
    print(f"{'network':28s} {'accuracy':>8s} {'ARC mistakes reproduced':>23s} {'(2nd > 0)':>9s} | near ties, share naming the third: " + ", ".join(G))
    for nm in models:
        print(f"{nm:28s} {100*st[(nm,'acc')]/st['all']:7.2f}% {100*st[(nm,'rep')]/st['w']:22.1f}% {100*st[(nm,'repp')]/st['wp']:8.1f}% | " + "  ".join(f"{100*st[(nm, g)]/st[(g, 'p')]:5.1f}%" for g in G))

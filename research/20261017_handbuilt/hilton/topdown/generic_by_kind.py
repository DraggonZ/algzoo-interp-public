"""Where do trained networks share more of ARC's mistakes than a generic imprecise solver? By kind of ARC mistake,
compare: the five other trained networks at 95-96.3% accuracy (seeds 103, 111, 115, 209, 210), a perfect solver
with fresh reading errors SD 0.022 (about 95.2% on the grid) and with fixed per-position offsets SD 0.03, and v18
(hand-built v18). Grid 40 x 48 x 48."""
import numpy as np, torch
import store_ladder as sl
from algzoo.core import load
from algzoo.handbuilt_v18 import build_v18
from mistakes_other_nets import T, net
DEV = "cuda"; dt = torch.float64; ND = 8
if __name__ == "__main__":
    g = torch.Generator(device=DEV).manual_seed(4); OFF = torch.randn(ND, 10, device=DEV, dtype=dt, generator=g) * 0.03
    tr = [net(T(load(f"/data/output/research/algzoo/20261016_run3/models/2nd_argmax_16_10_1073741824_{s}.pth"))) for s in (103, 111, 115, 209, 210)]
    v18 = net(T(build_v18().params))
    def fresh(X): return np.mean([(X + 0.022 * torch.randn(X.shape, device=DEV, dtype=dt, generator=g)).topk(2, 1).indices[:, 1] for _ in range(ND)], 0)
    names = ["trained 95-96%", "fresh errors 0.022", "position offsets 0.03", "v18 (96%)"]
    nV = len(sl.wtt); K, G, R, Wt = [], [], [], []; acc = np.zeros(4); Z = 0.0
    for i in range(0, 720, 16):
        tt = sl.full_mean.T[i:i + 16]; pos = tt.repeat_interleave(nV, 0); val = sl.Vt.repeat(len(tt), 1); n = len(pos)
        bmu = -torch.exp(-val[:, 2] ** 2 / 2) / np.sqrt(2 * np.pi) / (0.5 * (1 + torch.erf(val[:, 2] / np.sqrt(2))))
        X = bmu[:, None].repeat(1, 10); ar = torch.arange(n, device=DEV)
        for k in range(3): X[ar, pos[:, k]] = val[:, k]
        w = sl.wtt.repeat(len(tt)); y = pos[:, 1]; a0 = sl.run(X).argmax(1); wr = a0 != y
        same = [torch.stack([(f(X).argmax(1) == a0).double() for f in tr]).mean(0),
                torch.stack([((X + 0.022 * torch.randn(X.shape, device=DEV, dtype=dt, generator=g)).topk(2, 1).indices[:, 1] == a0).double() for _ in range(ND)]).mean(0),
                torch.stack([((X + OFF[d]).topk(2, 1).indices[:, 1] == a0).double() for d in range(ND)]).mean(0),
                (v18(X).argmax(1) == a0).double()]
        right = [torch.stack([(f(X).argmax(1) == y).double() for f in tr]).mean(0),
                 None, torch.stack([((X + OFF[d]).topk(2, 1).indices[:, 1] == y).double() for d in range(ND)]).mean(0), (v18(X).argmax(1) == y).double()]
        right[1] = torch.stack([((X + 0.022 * torch.randn(X.shape, device=DEV, dtype=dt, generator=g)).topk(2, 1).indices[:, 1] == y).double() for _ in range(ND)]).mean(0)
        acc += [(w * r).sum().item() for r in right]; Z += w.sum().item()
        kind = torch.where(a0 == pos[:, 0], 0, torch.where(a0 == pos[:, 2], 1, 2))
        K.append(kind[wr].cpu().numpy()); G.append(((val[:, 1] - val[:, 2]) / val[:, 1].abs())[wr].cpu().numpy())
        R.append(torch.stack([s[wr] for s in same], 1).cpu().numpy()); Wt.append(w[wr].cpu().numpy())
    K, G, R, Wt = map(np.concatenate, (K, G, R, Wt))
    print("grid accuracy: " + ", ".join(f"{nm} {100*a/Z:.2f}%" for nm, a in zip(names, acc)))
    print(f"{'ARC mistake kind':42s} {'share':>6s} " + " ".join(f"{nm:>22s}" for nm in names))
    def row(nm, m): print(f"{nm:42s} {100*Wt[m].sum()/Wt.sum():5.1f}% " + " ".join(f"{100*(Wt*R[:, j])[m].sum()/Wt[m].sum():21.1f}%" for j in range(4)))
    row("all", K >= 0)
    for k, nm in enumerate(("names the largest", "names the third", "names another")): row(nm, K == k)
    for lo, hi in ((-9, 0.03), (0.03, 0.1), (0.1, 9)): row(f"  names third, 2nd-3rd gap {lo:.2f}-{hi:.2f} of 2nd", (K == 1) & (G >= lo) & (G < hi))

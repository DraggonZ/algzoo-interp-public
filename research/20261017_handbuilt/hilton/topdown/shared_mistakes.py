"""Which of ARC's mistakes do the 32 other trained networks share? On ARC's wrong grid lists: what ARC names (the
largest, the third, another), the gap between the second and third numbers, and the share of the 32 networks (and the
first twin) making the same wrong choice, weighted by probability."""
import numpy as np, torch
import store_ladder as sl
from algzoo.core import load
from mistakes_other_nets import T, net
DEV = "cuda"; dt = torch.float64
if __name__ == "__main__":
    d = np.load("../../bridge/v18ts_matched2.npz"); twin = net(T([d[k] for k in ("wi", "wh", "wo")]))
    seeds = list(range(100, 116)) + list(range(200, 216))
    nets = [net(T(load(f"/data/output/research/algzoo/20261016_run3/models/2nd_argmax_16_10_1073741824_{s}.pth"))) for s in seeds]
    nV = len(sl.wtt); K, G, Sh, Tw, Wt = [], [], [], [], []
    for i in range(0, 720, 16):
        tt = sl.full_mean.T[i:i + 16]; pos = tt.repeat_interleave(nV, 0); val = sl.Vt.repeat(len(tt), 1); n = len(pos)
        bmu = -torch.exp(-val[:, 2] ** 2 / 2) / np.sqrt(2 * np.pi) / (0.5 * (1 + torch.erf(val[:, 2] / np.sqrt(2))))
        X = bmu[:, None].repeat(1, 10); ar = torch.arange(n, device=DEV)
        for k in range(3): X[ar, pos[:, k]] = val[:, k]
        w = sl.wtt.repeat(len(tt)); y = pos[:, 1]; a0 = sl.run(X).argmax(1); wr = a0 != y
        kind = torch.where(a0 == pos[:, 0], 0, torch.where(a0 == pos[:, 2], 1, 2))
        K.append(kind[wr].cpu().numpy()); G.append(((val[:, 1] - val[:, 2]) / val[:, 1].abs())[wr].cpu().numpy())
        Sh.append(torch.stack([(f(X).argmax(1) == a0)[wr] for f in nets], 1).double().mean(1).cpu().numpy())
        Tw.append((twin(X).argmax(1) == a0)[wr].cpu().numpy()); Wt.append(w[wr].cpu().numpy())
    K, G, Sh, Tw, Wt = map(np.concatenate, (K, G, Sh, Tw, Wt))
    def row(name, m):
        print(f"{name:40s} {100*Wt[m].sum()/Wt.sum():5.1f}% of ARC's mistakes | other nets share {100*(Wt*Sh)[m].sum()/Wt[m].sum():5.1f}% | twin {100*(Wt*Tw)[m].sum()/Wt[m].sum():5.1f}%")
    row("all", K >= 0)
    for k, nm in enumerate(("ARC names the largest", "ARC names the third", "ARC names another")): row(nm, K == k)
    for lo, hi in ((-9, 0.03), (0.03, 0.1), (0.1, 9)): row(f"  names third, gap 2nd-3rd {lo:.2f}-{hi:.2f} of 2nd", (K == 1) & (G >= lo) & (G < hi))
    for q in (0.5, 0.75): row(f"lists where >= {int(q*100)}% of other nets agree", Sh >= q)

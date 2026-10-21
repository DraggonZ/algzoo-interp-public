"""Development look (not counted as cost): the tie-band widths on the reference grid. For each triple and each (u, b),
w23 = the gap a at which the network first becomes right, starting from the 2nd-3rd tie (0 if right at the tie; the
grid's largest a, 5, if never); w12 likewise along b from the 1st-2nd tie at each (u, a). Saves widths_<net>.npz and
prints a few triples' width tables to see how widths depend on the third-largest u and the other gap."""
import sys, numpy as np, torch
sys.path.insert(0, "/data/output/research/algzoo/20261017_handbuilt/hilton/propagate")
import full_mean
import common as c
log = c.Log("widths.log")
nu, na = 56, 96
V, wt = full_mean.grid(nu, na); Vt = torch.tensor(V, device=c.DEV, dtype=c.dt)
u = V[:, 2].reshape(nu, na, na)[:, 0, 0]; g = (V[:, 1] - V[:, 2]).reshape(nu, na, na)[0, :, 0]
gt = torch.tensor(np.r_[0.0, g], device=c.DEV, dtype=c.dt)
for name, W in c.nets():
    if name not in (sys.argv[1:] or ["ARC"]): continue
    w23 = np.zeros((720, nu, na)); w12 = np.zeros((720, nu, na))
    for i in range(0, 720, 3):
        tt = c.T[i:i + 3]; k = len(tt)
        pos = tt.repeat_interleave(len(Vt), 0); val = Vt.repeat(k, 1)
        S, _ = c.scores(W, c.inputs(pos, val)); m, _, _ = c.margin_from(S, pos[:, 1])
        wr = (m <= 0).reshape(k, nu, na, na).to(torch.int8)
        n23 = torch.cumprod(wr, 2).sum(2)          # [k, u, b]: number of wrong a-cells from the tie
        n12 = torch.cumprod(wr, 3).sum(3)          # [k, u, a]
        # width = grid a of the first right cell, minus half a cell: use the edge between cells
        edges = torch.cat([torch.zeros(1, device=c.DEV, dtype=c.dt), (gt[1:-1] + gt[2:]) / 2, gt[-1:]])
        w23[i:i + k] = edges[n23.long()].cpu().numpy(); w12[i:i + k] = edges[n12.long()].cpu().numpy()
    np.savez(f"widths_{name}.npz", w23=w23, w12=w12, u=u, g=g)
    log(f"== {name}; u columns:", " ".join(f"{x:5.2f}" for x in u[::6]))
    for ti in (0, 100, 250, 400, 555, 700):
        log(f"triple {tuple(c.T[ti].tolist())}  w23 rows: b = ", " ".join(f"{g[j]:.2f}" for j in (10, 25, 40, 55)))
        for j in (10, 25, 40, 55): log("   ", " ".join(f"{x:5.3f}" for x in w23[ti, ::6, j]))
        log(f"   w12 rows: a = same")
        for j in (10, 25, 40, 55): log("   ", " ".join(f"{x:5.3f}" for x in w12[ti, ::6, j]))

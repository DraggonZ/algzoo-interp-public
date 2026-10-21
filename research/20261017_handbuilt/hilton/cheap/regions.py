"""Development look (not counted as cost): where the reduction's error mass lies by the sign of the second- and
third-largest. (A) second-largest below 0; (B) second-largest above 0, third below 0; (C) third above 0, split into
the band at the 2nd-3rd tie, the band at the 1st-2nd tie, both, and the rest (as in ref_triples.py)."""
import sys, numpy as np, torch
sys.path.insert(0, "/data/output/research/algzoo/20261017_handbuilt/hilton/propagate")
import full_mean
import common as c
log = c.Log("regions.log")
nu, na = 56, 96
V, wt = full_mean.grid(nu, na); Vt = torch.tensor(V, device=c.DEV, dtype=c.dt)
wt = torch.tensor(wt, device=c.DEV, dtype=c.dt).reshape(nu, na, na); mass = wt.sum() * 720
m2 = Vt[:, 1].reshape(nu, na, na); m3 = Vt[:, 2].reshape(nu, na, na)
A = m2 < 0; B = (m2 >= 0) & (m3 < 0); C = m3 >= 0
log(f"probability mass: A {100*(wt*A).sum().item()*720/mass.item():.2f} points, B {100*(wt*B).sum().item()*720/mass.item():.2f}, C {100*(wt*C).sum().item()*720/mass.item():.2f}")
for name, W in c.nets():
    if name not in (sys.argv[1:] or ["ARC", "200", "208", "212"]): continue
    acc = torch.zeros(7, device=c.DEV, dtype=c.dt)
    for i in range(0, 720, 3):
        tt = c.T[i:i + 3]; k = len(tt)
        pos = tt.repeat_interleave(len(Vt), 0); val = Vt.repeat(k, 1)
        S, _ = c.scores(W, c.inputs(pos, val)); m, _, _ = c.margin_from(S, pos[:, 1])
        wr = (m <= 0).reshape(k, nu, na, na); wi8 = wr.to(torch.int8)
        b23 = torch.cumprod(wi8, 2).bool(); b12 = torch.cumprod(wi8, 3).bool()
        for j, msk in enumerate([wr & A, wr & B, wr & C & b23 & ~b12, wr & C & b12 & ~b23, wr & C & b12 & b23, wr & C & ~b12 & ~b23, ~wr & A]):
            acc[j] += (msk.to(c.dt) * wt).sum()
    e = (100 * acc / mass).cpu().numpy()
    log(f"{name:>4s}: error points A {e[0]:.2f} (right in A: {e[6]:.3f}), B {e[1]:.2f}, C: band23 {e[2]:.2f} band12 {e[3]:.2f} both {e[4]:.2f} rest {e[5]:.2f}; total {e[:6].sum():.2f}")

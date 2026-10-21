"""Development look: inside the region where the second-largest is below 0 (mass 1.07 points), where is the
converged reduction right? Share of the region's mass that is right, by bins of m2, on ARC and three development nets."""
import sys, numpy as np, torch
import common as c, bandlib as bl
log = c.Log("low_region.log")
bins = [-9, -1.5, -1.0, -0.6, -0.3, -0.15, 0.0]
M2 = bl.U + bl.GA
for name, W in c.nets():
    if name not in (sys.argv[1:] or ["ARC", "200", "208", "212"]): continue
    right = torch.zeros(len(bins) - 1, device=c.DEV, dtype=c.dt); tot = torch.zeros_like(right)
    Vt = torch.stack([(bl.U + bl.GA + bl.GB).ravel(), M2.ravel(), bl.U.ravel()], 1)
    for i in range(0, 720, 3):
        tt = c.T[i:i + 3]; k = len(tt); pos = tt.repeat_interleave(len(Vt), 0); val = Vt.repeat(k, 1)
        S, _ = c.scores(W, c.inputs(pos, val)); m, _, _ = c.margin_from(S, pos[:, 1]); r = (m > 0).reshape(k, *bl.U.shape).to(c.dt)
        for j in range(len(bins) - 1):
            msk = ((M2 >= bins[j]) & (M2 < bins[j + 1])).to(c.dt) * bl.WT
            right[j] += (r * msk).sum(); tot[j] += msk.sum() * k
    log(f"{name:>4s}: share right by m2 bin " + "; ".join(f"[{bins[j]},{bins[j+1]}): {right[j]/tot[j]:.2f} (mass {100*tot[j].item()/bl.MASS:.3f} pts)" for j in range(len(bins) - 1)))

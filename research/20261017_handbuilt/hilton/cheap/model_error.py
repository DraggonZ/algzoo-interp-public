"""Development: where does the band model (exact node widths, 'norm' law) differ from the converged reduction? For
each network, the model's accuracy minus the converged reduction's, split by region of (u, a, b): second-largest
below 0; second-largest above 0 with a < b (nearer the 2nd-3rd tie); a >= b (nearer the 1st-2nd tie). Each region's
share is (reduction wrong mass - model wrong mass) there, in points."""
import sys, numpy as np, torch
import common as c, bandlib as bl
log = c.Log("model_error.log")
S23 = bl.scale("23", "norm"); S12 = bl.scale("12", "norm"); low = (bl.U + bl.GA) < 0
regs = {"m2<0": low, "a<b": (~low) & (bl.GA < bl.GB), "a>=b": (~low) & (bl.GA >= bl.GB)}
Vt = torch.stack([(bl.U + bl.GA + bl.GB).ravel(), (bl.U + bl.GA).ravel(), bl.U.ravel()], 1)
names = sys.argv[1:] or ["ARC"] + [str(s) for s in c.SEEDS_DEV]
tot = []
for name, W in c.nets():
    if name not in names: continue
    d = np.load(f"path_widths_{name}.npz"); A = torch.tensor(d["A"], device=c.DEV); B = torch.tensor(d["B"], device=c.DEV)
    acc = {k: 0.0 for k in regs}; per = np.zeros(720)
    for i in range(0, 720, 3):
        tt = c.T[i:i + 3]; k = len(tt); pos = tt.repeat_interleave(len(Vt), 0); val = Vt.repeat(k, 1)
        S, _ = c.scores(W, c.inputs(pos, val)); m, _, _ = c.margin_from(S, pos[:, 1]); wr = (m <= 0).reshape(k, *bl.U.shape)
        for j in range(k):
            t = i + j; wm = (bl.GA < A[t] * S23) | (bl.GB < B[t] * S12) | low
            diff = (wr[j].to(c.dt) - wm.to(c.dt)) * bl.WT
            per[t] = 100 * diff.sum().item() / bl.MASS
            for kk, msk in regs.items(): acc[kk] += 100 * (diff * msk).sum().item() / bl.MASS
    np.save(f"model_error_{name}.npy", per)
    srt = np.sort(per); log(f"{name:>4s}: model minus reduction {sum(acc.values()):+.2f} points = " + ", ".join(f"{k} {v:+.2f}" for k, v in acc.items())
        + f"; per triple: 10 most negative sum {srt[:10].sum():+.2f}, 10 most positive {srt[-10:].sum():+.2f}")
    tot.append(list(acc.values()))
t = np.array(tot); log("RMSE over networks of each region's part: " + ", ".join(f"{k} {np.sqrt((t[:,j]**2).mean()):.2f}" for j, k in enumerate(regs)))

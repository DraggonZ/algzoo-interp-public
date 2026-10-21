"""Reference (development only, not counted as cost): the converged mean-background reduction per position triple, on
the grid of propagate/full_mean.py (56 x 96 x 96), with the wrong mass split into
  band23: wrong points connected to the 2nd-3rd tie (a = 0) along a (wrong at the smallest a and every a below),
  band12: the same along b from the 1st-2nd tie,  both: in both,  rest: wrong in neither band ('bulk').
Saves ref_triples_<net>.npz. Networks: ARC and the development seeds 200-215."""
import sys, time, numpy as np, torch
sys.path.insert(0, "/data/output/research/algzoo/20261017_handbuilt/hilton/propagate")
import full_mean
import common as c
log = c.Log("ref_triples.log")
V, wt = full_mean.grid(56, 96); nu, na = 56, 96
Vt = torch.tensor(V, device=c.DEV, dtype=c.dt); wt = torch.tensor(wt, device=c.DEV, dtype=c.dt).reshape(nu, na, na)
mass = wt.sum().item() * 720
names = sys.argv[1:] or ["ARC"] + [str(s) for s in c.SEEDS_DEV]
for name, W in c.nets():
    if name not in names: continue
    t0 = time.time(); out = np.zeros((720, 4)); per = 3
    for i in range(0, 720, per):
        tt = c.T[i:i + per]; k = len(tt)
        pos = tt.repeat_interleave(len(Vt), 0); val = Vt.repeat(k, 1)
        S, _ = c.scores(W, c.inputs(pos, val)); m, _, _ = c.margin_from(S, pos[:, 1])
        wr = (m <= 0).reshape(k, nu, na, na)                                 # [k, u, a, b]
        b23 = torch.cumprod(wr.to(torch.int8), dim=2).bool()                 # wrong from a=0 up to here
        b12 = torch.cumprod(wr.to(torch.int8), dim=3).bool()
        for j, (msk) in enumerate([b23 & ~b12, b12 & ~b23, b23 & b12, wr & ~b23 & ~b12]):
            out[i:i + k, j] = (msk.to(c.dt) * wt).sum((1, 2, 3)).cpu().numpy() / mass
    np.savez(f"ref_triples_{name}.npz", err=out)
    tot = out.sum(0) * 100
    log(f"{name:>4s}: reduction accuracy {100 - tot.sum():.3f}%; error points band23 {tot[0]:.2f}, band12 {tot[1]:.2f}, both {tot[2]:.2f}, rest {tot[3]:.2f}  {time.time()-t0:.0f}s")

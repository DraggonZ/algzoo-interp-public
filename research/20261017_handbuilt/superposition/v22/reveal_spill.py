"""v22 step 2: the revealed record's spill. In ARC an overtaken record's wave spills onto other positions (waves.log,
'all neurons, weight of P': about +-0.5-1.5 votes per unit); v7z trims it to a spike by dipping the memory by w times
the old largest one step after the overtaking. Less trim leaves more of the wave. Sweep w (v7z's own parameter, 0.8 by
default), with the hiding leftover of hide_leftover.py recomputed for each; report the P table's match to ARC
(correlation over the table) and the scores."""
import sys, numpy as np, torch
sys.path.insert(0, "../../family"); sys.path.insert(0, "..")
from common import *
import score
from baseline_v7z import vz
import hide_leftover as hl
from waves import group_jac
def v7z(w):
    vz.set_slots(9); W = T(vz.build_v7(clock="z", w=w).params); vz.set_slots(8); return W
def p_table(W, X):
    n = len(X); m = torch.zeros(n, device=DEV, dtype=dt); Lp = torch.full((n,), -1, device=DEV, dtype=torch.long); Pp = Lp.clone()
    for t in range(10):
        new = X[:, t] > m; Pp = torch.where(new, Lp, Pp); Lp = torch.where(new, t, Lp); m = torch.where(new, X[:, t], m)
    J = group_jac(W, X, list(range(len(W[0])))); kept = (X > 0) & (torch.arange(10, device=DEV)[None] != Lp[:, None])
    vote = J[:, torch.arange(10), torch.arange(10)][kept].median(); Tb = torch.zeros(10, 10, device=DEV, dtype=dt)
    for r in range(1, 9):
        mm = (Lp == r) & (Pp >= 0)
        if mm.sum() > 100: Tb[r] = (J[mm][torch.arange(mm.sum().item(), device=DEV), :, Pp[mm]] / vote).median(0).values
    return Tb
if __name__ == "__main__":
    X = torch.randn(100000, 10, device=DEV, dtype=dt, generator=torch.Generator(device=DEV).manual_seed(3))
    Pa = p_table(ARC, X)[1:9]; models = {}
    for w in (0.8, 0.6, 0.4, 0.2):
        hl.nets = lambda w=w: {"v7 + z + ninth slot (18)": v7z(w)}
        Wn, r2, _ = hl.build(); Pv = p_table(Wn, X)[1:9]
        c = np.corrcoef(Pa.flatten().cpu().numpy(), Pv.flatten().cpu().numpy())[0, 1]
        print(f"w {w}: P-table correlation with ARC {c:.2f}; RMS ARC {Pa.pow(2).mean().sqrt():.2f}, this {Pv.pow(2).mean().sqrt():.2f}; hiding leftover match {100*r2:.0f}%")
        models[f"v7z w {w} + hiding leftover"] = [Wn]
    score.report(models)

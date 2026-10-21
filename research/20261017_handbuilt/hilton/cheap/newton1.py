"""Band widths at the node from the tie run plus one forward tangent (2 runs per tie input with a loser): width =
-D / slope, the zero of the margin on the tie's own linear piece; slope <= 0 means the margin falls as the gap
opens, read as 'wrong over the whole gap' (width 5, the grid's end). Compared with exact node widths (fine scan) and
with the converged reduction, on ARC and the development seeds."""
import sys, numpy as np, torch
import common as c, bandlib as bl
log = c.Log("newton1.log")
names = sys.argv[1:] or ["ARC"] + [str(s) for s in c.SEEDS_DEV]
allnames = [n for n, _ in c.nets()]
rows = []
for name, W in c.nets():
    if name not in names: continue
    ex = {t: bl.exact_width(W, t, c.T) for t in ("23", "12")}
    np.savez(f"exact_node_{name}.npz", A=ex["23"].cpu().numpy(), B=ex["12"].cpu().numpy())
    wN = {}
    for tie in ("23", "12"):
        X, d = bl.node_inputs(tie, c.T); S, dS = c.scores(W, X, d); D, sl, k = c.margin_from(S, c.T[:, 1], dS)
        wN[tie] = torch.where(D >= 0, 0.0, torch.where(sl > 0, (-D / sl.clamp_min(1e-12)).clamp_max(5.0), 5.0))
    ref = 100 - 100 * np.load(f"ref_triples_{name}.npz")["err"].sum() if name != "x" else 0
    e_ex = bl.accuracy(ex["23"], ex["12"]); e_n = bl.accuracy(wN["23"], wN["12"])
    tru = 100 * c.TRUE[allnames.index(name)]
    rows.append((ref, e_ex, e_n, tru))
    log(f"{name:>4s}: true {tru:.2f}; converged reduction {ref:.2f}; band model with exact node widths {e_ex:.2f}; with one Newton step {e_n:.2f}")
r = np.array(rows)
for j, nm in ((1, "exact node widths"), (2, "one Newton step")):
    log(f"{nm}: RMSE against converged reduction {np.sqrt(((r[:, j] - r[:, 0])**2).mean()):.2f}, against truth {np.sqrt(((r[:, j] - r[:, 3])**2).mean()):.2f} points ({len(r)} networks)")
log(f"converged reduction against truth: RMSE {np.sqrt(((r[:, 0] - r[:, 3])**2).mean()):.2f}")

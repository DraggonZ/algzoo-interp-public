"""Development: band widths from the tie margin alone, width = max(-D, 0) / slope, with one slope per network and tie
(here the 'oracle' slope: sum of -D over the losers divided by the sum of the exact node widths). How much does the
model lose against exact node widths (exact_node.py)? Also the share of losers whose winner at the tie is a third position."""
import sys, numpy as np, torch
import common as c, bandlib as bl
log = c.Log("donly.log")
names = sys.argv[1:] or ["ARC", "200", "208", "212"]
for name, W in c.nets():
    if name not in names: continue
    e = np.load(f"exact_node_{name}.npz"); ref = 100 - 100 * np.load(f"ref_triples_{name}.npz")["err"].sum()
    out = []
    WD = {}
    for tie, ex in (("23", e["A"]), ("12", e["B"])):
        X, d = bl.node_inputs(tie, c.T); S, dS = c.scores(W, X, d); D, sl, k = c.margin_from(S, c.T[:, 1], dS)
        D = D.cpu().numpy(); sl = sl.cpu().numpy(); k = k.cpu().numpy(); los = D < 0
        other = c.T[:, 2 if tie == "23" else 0].cpu().numpy()
        sig = (-D[los]).sum() / ex[los].sum()
        lin = np.where(los, -D / np.where(sl > 1, sl, np.median(sl[los])), 0)
        WD[tie] = (np.where(los, -D / sig, 0), lin)
        out.append(f"tie {tie}: losers {los.sum()}, winner a third position {(los & (k != other)).sum()}; oracle slope {sig:.1f} (median tangent {np.median(sl[los]):.1f}); "
                   f"corr(width, -D) {np.corrcoef(ex[los], -D[los])[0,1]:.2f}, corr(width, -D/tangent) {np.corrcoef(ex[los], lin[los])[0,1]:.2f}")
    t = lambda x: torch.tensor(x, device=c.DEV, dtype=c.dt)
    log(f"{name:>4s}: converged {ref:.2f}; exact node widths {bl.accuracy(t(e['A']), t(e['B'])):.2f}; -D/oracle slope {bl.accuracy(t(WD['23'][0]), t(WD['12'][0])):.2f}; -D/own tangent {bl.accuracy(t(WD['23'][1]), t(WD['12'][1])):.2f}")
    for o in out: log("      " + o)

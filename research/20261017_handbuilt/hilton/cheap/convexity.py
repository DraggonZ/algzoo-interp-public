"""Is the margin convex along the gap near a tie (slope rising as the gap opens)? For ARC's triples wrong at a tie
(nodes of cheap_est.py), the margin's slope at the tie (right after it) and at a = 0.05, 0.1 and 0.37 (a0); and how
often a straight line between the tie and a node at gap 0.044 or 0.1 (as in a coarse grid) misses the band."""
import numpy as np, torch
import common as c, cheap_est as ce, pathband as pb
log = c.Log("convexity.log")
W = c.nets()[0][1]
for tie in ("23", "12"):
    nd = ce.N23 if tie == "23" else ce.N12
    X0 = ce._X(c.T, nd); S, _ = c.scores(W, X0); D, _, _ = c.margin_from(S, c.T[:, 1]); li = (D < 0).nonzero().squeeze(1)
    P = c.T[li]; n = len(P); ar = torch.arange(n, device=c.DEV); X0 = ce._X(P, nd); d = torch.zeros_like(X0); d[ar, P[:, 0]] = 1
    if tie == "23": d[ar, P[:, 1]] = 1
    sl = {}; mv = {}
    for g in (1e-6, 0.044, 0.05, 0.1, 0.37):
        S, dS = c.scores(W, X0 + g * d, d); m, s, _ = c.margin_from(S, P[:, 1], dS); sl[g] = s.cpu().numpy(); mv[g] = m.cpu().numpy()
    w, _, _ = pb.widths(W, tie, P, nd); w = w.cpu().numpy(); D0 = D[li].cpu().numpy()
    log(f"tie {tie}: {n} wrong orderings; median margin slope at gap 0+, 0.05, 0.1, 0.37: " + ", ".join(f"{np.median(sl[g]):.0f}" for g in (1e-6, 0.05, 0.1, 0.37))
        + f"; share with slope at 0.05 above slope at 0+: {(sl[0.05] > sl[1e-6]).mean():.2f}")
    for g in (0.044, 0.1):
        lin = np.where(mv[g] > 0, g * (-D0) / (mv[g] - D0), np.nan); has = mv[g] > 0
        log(f"   straight line from the tie to gap {g}: right there for {has.mean():.2f} of them; its zero over the exact width, median {np.nanmedian(lin[has] / np.maximum(w[has], 1e-6)):.2f}; "
            f"sum of line zeros {np.nansum(lin[has]):.2f} against exact widths {w[has].sum():.2f}")

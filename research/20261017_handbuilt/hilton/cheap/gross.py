"""Development look: thin and wide bands. The band model with exact node widths, with only the thin bands (width < 0.3), and with no bands; how many wide bands each network has, and whether the tie runs alone flag them (a third position wins the tie) or the tangent does (slope <= 0)."""
import numpy as np, torch, sys
log = None
import common as c, bandlib as bl
LOG = c.Log("gross.log")
for name, W in c.nets():
    if name not in ["ARC", "200", "201", "208", "212", "215"]: continue
    e = np.load(f"exact_node_{name}.npz"); A, B = e["A"], e["B"]
    t = lambda x: torch.tensor(x, device=c.DEV, dtype=c.dt)
    full = bl.accuracy(t(A), t(B))
    thinA = np.where(A < 0.3, A, 0); thinB = np.where(B < 0.3, B, 0)
    thin = bl.accuracy(t(thinA), t(thinB))
    zero = bl.accuracy(t(0 * A), t(0 * B))
    feats = {}
    for tie in ("23", "12"):
        X, d = bl.node_inputs(tie, c.T); S, dS = c.scores(W, X, d); D, sl, k = c.margin_from(S, c.T[:, 1], dS)
        feats[tie] = (D.cpu().numpy(), sl.cpu().numpy(), k.cpu().numpy())
    T = c.T.cpu().numpy()
    # gross flags from the tie runs: at the a-tie the winner is not p2/p3; at the b-tie the winner is not p1/p2
    fa = (feats["23"][2] != T[:, 2]) & (feats["23"][0] < 0); fb = (feats["12"][2] != T[:, 0]) & (feats["12"][0] < 0)
    ga, gb = A >= 0.3, B >= 0.3
    LOG(f"{name}: band model full {full:.2f}, thin only {thin:.2f}, no bands {zero:.2f}; gross A {ga.sum()} B {gb.sum()}; "
          f"flag a-tie third-party winner {fa.sum()} (of which gross A {(fa&ga).sum()}), flag b-tie {fb.sum()} (gross B {(fb&gb).sum()}); gross A with slope<=0 {(ga & (feats['23'][1]<=0)).sum()}, thin A with slope<=0 {((~ga)&(A>0)&(feats['23'][1]<=0)).sum()}")

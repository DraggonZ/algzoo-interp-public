"""Development look: the widest bands on seed 208 (exact node widths) with their tie margin, tangent slope and the winner at the tie."""
import numpy as np, torch, sys
pass
import common as c, bandlib as bl
LOG = c.Log("wide_examples.log")
W = dict(c.nets())["208"]; e = np.load("exact_node_208.npz")
for tie, ex in (("23", e["A"]), ("12", e["B"])):
    X, d = bl.node_inputs(tie, c.T); S, dS = c.scores(W, X, d); D, sl, k = c.margin_from(S, c.T[:, 1], dS)
    D = D.cpu().numpy(); sl = sl.cpu().numpy(); k = k.cpu().numpy(); T = c.T.cpu().numpy(); los = D < 0
    LOG("tie", tie, "width quantiles (losers):", np.quantile(ex[los], [.1, .25, .5, .75, .9, .95, .99]).round(3), " share of width sum from widths > 0.3:", (ex[los & (ex > .3)].sum() / ex[los].sum()).round(2))
    big = np.argsort(-ex)[:12]
    for i in big: LOG("   ", T[i], "width %.3f D %.2f tangent %.1f winner %d" % (ex[i], D[i], sl[i], k[i]))
    # widths among right-at-tie
    LOG("   right at tie but width>0? ", (ex[~los] > 0).sum())

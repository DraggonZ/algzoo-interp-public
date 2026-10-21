"""Hand-built w12b (wider track): w8b_gapread's network (handbuilt_w8b) with the keep gate serving as the first vote
cell, so that the linear route has no reader (research/20261017_handbuilt/faithful_twin/LEDGER.md, branch w12b).

Rule (free_linear). In w8b the linear route L and the keep gate G are read only by the newest vote cell mem0 (-1 each)
and by position 8 of the readout (+1 each). From their rows, when the keep gate is open (the runner-up comparator Q is
0, i.e. the number is at or above 0.75 of the runner-up) and unclipped:
    L(n+1) = 3 G(n+1) + 0.225 S(n) + 0.0045 D(n) + c . clocks(n)       (c: L's zero-sum read of the clocks)
           = 3 G(n+1) + 0.225 S(n+1) + c R^+ clocks(n+1)                (S(n+1) = S(n) + 0.02 D(n); R the clocks' turn)
with R^+ the inverse of the clocks' turn on their zero-sum part (exact while no start or margin enters the clocks).
The coefficients (3, 0.225) are not typed in: they are solved from L's and G's rows (L's C7 weight over G's, and what is
left on S and D, checked to be proportional to the store's own row). Readers of L + G then read 4 G + 0.225 S + c R^+
clocks. A number below the cut votes 0 instead of its value; it is never the answer (two larger numbers precede it).
"""
import numpy as np
from algzoo.handbuilt_w8b import build_w8b, w8b_map

CLOCKS = ("clock0", "clock1", "clock2", "turn0", "turn1", "turn2")


def free_linear(wi, wh, wo, idx, pos8=True):
    wi, wh, wo = wi.copy(), wh.copy(), wo.copy()
    S, D, G, L, C7, Q = (idx[k] for k in ("store", "detect", "keep", "linear", "c7", "Q"))
    C = [idx[k] for k in CLOCKS]
    n = len(wi)
    # L's row = r G's row (on C7) + rest on S, D + clock read c
    r = wh[L, C7] / wh[G, C7]
    rest = wh[L].copy(); rest -= r * wh[G]; rest[Q] = 0.0   # G's Q term is 0 when the gate is open
    c = np.zeros(n); c[C] = rest[C]; rest[C] = 0.0
    assert set(np.flatnonzero(np.abs(rest) > 1e-12)) <= {S, D}, np.flatnonzero(np.abs(rest) > 1e-12)
    lam = rest[S] / wh[S, S]
    assert np.isclose(rest[D], lam * wh[S, D]), (rest[D], lam * wh[S, D])   # rest = lam x the store's own row
    # one step back through the clocks' turn (pseudo-inverse on each triple's zero-sum part)
    back = np.zeros(n)
    for tri in (C[:3], C[3:]):
        R = wh[np.ix_(tri, tri)]
        back[tri] = c[tri] @ np.linalg.pinv(R)
        back[tri] -= back[tri].mean()          # read only the zero-sum part (the level is the arriving number's)
    # substitute in every reader of L (neurons and readout): L -> r G + lam S + back . clocks
    for M in (wh, wo):
        w = M[:, L].copy()
        if M is wo and not pos8:
            continue
        M[:, G] += w * r
        M[:, S] += w * lam
        M[:, :] += np.outer(w, back)
        M[:, L] = 0.0
    wh[L, :] = 0.0; wi[L] = 0.0
    return wi, wh, wo, dict(r=r, lam=lam, back=back[C])


def build_w12b(**kw):
    wi, wh, wo, idx, _ = build_w8b(**kw)
    wi2, wh2, wo2, info = free_linear(wi, wh, wo, idx)
    return wi2, wh2, wo2, idx, info

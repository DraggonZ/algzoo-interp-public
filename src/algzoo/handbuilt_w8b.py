"""Hand-built w8b (wider track, 24 neurons): w7b_twopair's network (handbuilt_w7b) with the clocks reading the wave's
start and margin from three hiding comparators instead of from one hiding comparator minus C7
(research/20261017_handbuilt/faithful_twin/LEDGER.md, branch w8b).

Why. In w7b every clock (the 45-degree clocks in ARC slot 3 and the 135-degree clocks in slot 12) reads the start as
H7 - C7 and the margin from C1. C7 is no hiding neuron, so the clocks' input from the hiding slots is H7's whole
comparator level g (m1 - x), not a function of the gap (job 11 fails its use test).

The hiding comparators share C7's level g (m1(t) - x_{t+1}) and differ in references. Unclipped (the arriving number
below every reference), relative to C7, with d the jump of a record at step t (special parts, read off the rows):
    H7 - C7:  d at age 1 (one step after the record), 0 otherwise           (w6b: H7 is C7 plus c = 1 on the jump)
    C1 - C7:  -c1 g d at age 2, 0 otherwise                                  (c1 = 0.95, read off ARC in w4b)
    C6 - C7:  -g d at age 1 (C6 compares with the previous largest), -c6 g d at age 2   (c6 = 0.23)
Rule (rebuild_reads): each clock row's weights on (H7, C1, C6, C7), w, become (alpha, beta, gamma, 0) with
    alpha + beta + gamma = sum(w)                (same level weight; 0, the clocks are level-free),
    age-1 part and age-2 part equal to w's        (same start, same margin),
three equations, three unknowns. Unclipped, every clock receives exactly w7b's input. The solution has alpha of order
1e-4 (H7 is nearly unread) and the start carried by C1 - C6: one step after a record C1 - C6 = g (m1 - max(previous
largest, x)) = g x the gap, also when the arriving number is the new second-largest (it clips C6 only). That case,
and the margin's clipping points two steps after, are where the function differs from w7b.
MAP: w7b's. No neuron added or removed; no constant read off ARC beyond w7b's.
"""
import numpy as np
from algzoo.handbuilt_w7b import build_w7b, w7b_map


def special_parts(wi, wh, idx, a=1000.0, s=20.0):
    """Each comparator's pre-activation minus C7's, per unit jump d, at ages 1 and 2 after a record (unclipped), and its
    level (input weight and store weight), from the rows. State at the step read: age 1: S = s P, D = a d, LG = 0,
    CR = a d (the record's own detector copy); age 2: S = s (P + d), D = 0, LG = d, CR = 0. P = x = 0, d = 1."""
    S, D, LG, CR = idx["store"], idx["detect"], idx["late"], idx["cancel"]
    st1 = {S: 0.0, D: a, LG: 0.0, CR: a}; st2 = {S: s, D: 0.0, LG: 1.0, CR: 0.0}
    pre = lambda n, st: sum(wh[n, k] * v for k, v in st.items())
    C7 = idx["c7"]; out = {}
    for nm in ("h7", "c1", "c6", "c7"):
        n = idx[nm]
        # the other inputs the comparators read must be only these (and nothing at all from the clocks)
        assert set(np.flatnonzero(np.abs(wh[n]) > 0)) <= {S, D, LG, CR}, nm
        out[nm] = (pre(n, st1) - pre(C7, st1), pre(n, st2) - pre(C7, st2), wi[n], wh[n, S])
    return out


def rebuild_reads(wi, wh, idx, readers):
    sp = special_parts(wi, wh, idx); src = ["h7", "c1", "c6", "c7"]; cols = [idx[k] for k in src]
    for nm in ("h7", "c1", "c6"):
        assert np.isclose(sp[nm][2], sp["c7"][2]) and np.isclose(sp[nm][3], sp["c7"][3]), "levels differ"
    A = np.array([[1.0, 1.0, 1.0], [sp[k][0] for k in src[:3]], [sp[k][1] for k in src[:3]]])
    wh = wh.copy(); sol = {}
    for r in readers:
        w = wh[r, cols]
        tgt = np.array([w.sum(), sum(w[i] * sp[k][0] for i, k in enumerate(src)), sum(w[i] * sp[k][1] for i, k in enumerate(src))])
        abg = np.linalg.solve(A, tgt)
        wh[r, cols[:3]] = abg; wh[r, cols[3]] = 0.0; sol[r] = (w, abg)
    return wh, sol


def build_w8b(**kw):
    wi, wh, wo, idx = build_w7b(**kw)
    readers = [idx[f"clock{k}"] for k in range(3)] + [idx[f"turn{k}"] for k in range(3)]
    # among the neurons only the clocks read H7 (checked; the linear route's 1.7e-16 is rounding). The readout keeps its
    # read of H7 - C7 at positions 4 and 8 (the start of a record at the last step), unchanged.
    assert set(np.flatnonzero(np.abs(wh[:, idx["h7"]]) > 1e-12)) == set(readers)
    wh2, sol = rebuild_reads(wi, wh, idx, readers)
    return wi, wh2, wo, idx, sol


def w8b_map(wi, wh, wo, idx):
    """w7b's map (the map rescales by RMS activity on w7b's own network, so it is read off the w7b build)."""
    return w7b_map(*build_w7b())

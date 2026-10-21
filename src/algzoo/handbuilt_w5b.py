"""Hand-built w5b (wider track, up to 24 neurons): w4b_margin's network (handbuilt_w4b) with the hiding wave put back
into ARC's hiding slots, so that silencing the hiding slots removes the hiding (research/20261017_handbuilt/
faithful_twin/LEDGER.md, branch w5b).

Starting point. In w4b the four hiding-wave neurons (flip P, clocks C0-C2) are helpers in no ARC slot; each holds the
comparator level g (m1 - x) plus its part of the wave. Readers take the level they need from C7 (comparator with the
largest, ARC slot 7). But the readers of the flip (linear route, readout rows 1-8) still read the flip's level and
cancel it with C7 (flip -0.765, C7 +0.765). So the arriving number x reaches the memory only through neurons that also
carry the wave: silence the wave neurons for one step and the next position's vote reads x = largest
(LEDGER w5b-a: 89% of lists then name the position after the largest).

Rule added here (option level_free=True): every reader outside the wave takes only the wave from the four wave
neurons. Its weights on them are changed by lambda (1, 1, 1, 0) on (C0, C1, C2, P), lambda = -(sum of its weights on
the four) / 3, and C7 takes the level it loses: + (that sum) on C7. The clocks' wave parts sum to zero, so the wave a
reader takes is unchanged; the levels of the four now cancel among themselves. Exact while no wave neuron clips (the
same condition as w4b's level rule). Then silencing the four wave neurons removes the wave and nothing else.

MAP (option wave_slot=7): ARC's neuron 7 compares with the largest at both delays (weights 0.00 and 0.00 on the
previous largest), the same reference as the four wave neurons' level; so ARC slot 7 is carried by the four wave
neurons together (equal weights; their sum is 4 x the level plus the flip's wave, the clocks' wave parts cancel in
the sum). C7 becomes a helper in no slot (it holds the arriving number for the memory, keep gate and runner-up
comparator, which in ARC read the input directly). Slots 1, 6, 8 keep C1, C6, Q as in w4b.

Option copy_share=r (w5b-3, 24 neurons): three helpers U0-U2 whose rows equal the clocks' rows (same inputs, same
weights), so U equals the clocks at every step; each clock then takes r of its rotation from U (clock-clock block
(1 - r) Rc, clock-U block r Rc). Same function exactly. A nudge of the hiding slots leaves (1 - r) of it in the clocks
one step later (job 8's test); silencing them for one step leaves U's copy, of which about r / (1 + r) comes back.
This is a split of one rotation over seven neurons, not ARC's rebuild of the hiding neurons from store, gap and late
gate (LEDGER w5b-3).

Option readout_level (w5b-2, dropped): see build_w5b.
"""
import numpy as np
from algzoo.handbuilt_w4b import build_w4b, arc_map

WAVE = ["clock0", "clock1", "clock2", "flip"]


def build_w5b(level_free=True, readout_level=False, copy_share=0.0, **kw):
    """readout_level=True (w5b-2): the readout takes the level it needs (position 9's vote of the last number, position
    0's runner-up read, the margin of a record at step 7) from the three clocks (slot 7) instead of the helper C7: each
    row's C7 weight is split equally over the clocks (a clock's level equals C7's while the clock is on; the clocks'
    wave parts sum to zero, so the wave read is unchanged). This is the twin's own readout (v23: position 9 reads the
    clock's level). Recurrent readers stay on C7, so silencing the hiding slots still leaves the arriving number."""
    kw = {**dict(late=True, level=True, refs=True, cancel=True, margin=True), **kw}
    wi, wh, wo, idx = build_w4b(**kw)
    wi, wh, wo = wi.copy(), wh.copy(), wo.copy()
    Wn = [idx[k] for k in WAVE]; C7 = idx["c7"]
    if level_free:
        def fix(row):
            s = row[Wn].sum(); row[Wn[:3]] -= s / 3; row[C7] += s
        for r in range(len(wi)):
            if r in Wn: continue                      # the wave's own rotation is already level-free
            if np.abs(wh[r, Wn]).sum() > 0: fix(wh[r])
        for p in range(10): fix(wo[p])
    if readout_level:
        for p in range(10):
            wo[p, Wn[:3]] += wo[p, C7] / 3; wo[p, C7] = 0.0
    if copy_share > 0:
        # U0-U2: rows equal to the clocks' rows (computed before the clocks' rows change), so U = clocks at every step.
        # Each clock then takes copy_share of its rotation from U: same function exactly.
        Cs = Wn[:3]; n0 = len(wi); n = n0 + 3; Us = list(range(n0, n))
        wi2 = np.zeros(n); wh2 = np.zeros((n, n)); wo2 = np.zeros((10, n))
        wi2[:n0] = wi; wh2[:n0, :n0] = wh; wo2[:, :n0] = wo
        for c, u in zip(Cs, Us):
            wi2[u] = wi[c]; wh2[u, :n0] = wh[c]; idx[f"copy{c - Cs[0]}"] = u
        Rc = wh[np.ix_(Cs, Cs)].copy()
        assert np.abs(wh[np.ix_(Cs, [idx["flip"]])]).max() == 0
        wh2[np.ix_(Cs, Cs)] = (1 - copy_share) * Rc; wh2[np.ix_(Cs, Us)] = copy_share * Rc
        wi, wh, wo = wi2, wh2, wo2
    return wi, wh, wo, idx


def w5b_map(wi, wh, wo, idx, wave_slot=7, n_lists=20000, seed=0):
    """w4b's map, then ARC slot `wave_slot` is carried by the four wave neurons (equal weights, RMS scale of their sum
    against ARC's neuron), and the neuron w4b put there becomes a helper."""
    import torch as th
    from algzoo.core import example
    M, lay = arc_map(wi, wh, wo, idx, n_lists, seed)
    M = M.copy(); M[wave_slot] = 0
    x = th.randn(n_lists, 10, generator=th.Generator().manual_seed(seed), dtype=th.float64).numpy()
    def acts(Wi, Wh):
        h = np.zeros((len(x), len(Wi))); out = []
        for t in range(10):
            h = np.maximum(h @ Wh.T + x[:, t:t + 1] * Wi, 0); out.append(h)
        return np.stack(out[1:], 1).reshape(-1, len(Wi))
    A = [np.asarray(p, float) for p in example()]
    a = acts(A[0], A[1])[:, wave_slot]; s = acts(wi, wh)[:, [idx[k] for k in WAVE]].sum(1)
    for k in WAVE: M[wave_slot, idx[k]] = np.sqrt((a ** 2).mean()) / np.sqrt((s ** 2).mean())
    lay = dict(lay); lay[wave_slot] = "+".join(WAVE)
    return M, lay

"""Hand-built w6b (wider track, up to 24 neurons): w5b_slot7's network (handbuilt_w5b, 21 neurons) with the hiding wave's
rotation held in neurons mapped to the memory, and ARC's hiding slot 7 given to a comparator that is rebuilt every step
and only relays a new record's wave start for one step (research/20261017_handbuilt/faithful_twin/LEDGER.md, branch
w6b).

Starting point. In w5b_slot7 the four hiding-wave neurons (clock0-2, flip; ARC slot 7) each hold the comparator level
g relu(m1(t-1) - x_t) plus their part of the hiding wave z_t (a 45-degree turning pair in the clocks, a sign flip in
the flip neuron). The wave turns on their own rotation: z_{t+1} = B z_t + s0 d_t (B: their block of weights on
themselves; s0 d_t: the start written by the detector one step after a record, d_t the record's jump). So a nudge of
slot 7 is kept whole in the readout's view (job 8 fails on that view, 0.86 against the 0.8 bar; the slot-7 part alone
is 0.71, `w6b/a1_job8_parts.log`).

Rule here (build_w6b):
  H7, a new comparator in ARC slot 7: H7_{t+1} = relu(g (m1(t) - x_{t+1}) + c d_t), i.e. C7's row plus c / a on the
      detector. While x_{t+1} is below the largest, H7 - C7 = c d_t: the jump of a record one step before, else 0. A
      record at t+1 clips H7 and C7 together (their level is g times minus the amount beaten), so a new record wipes the
      relayed start, as it wipes the old wave. H7 reads no hiding neuron and not itself: rebuilt every step from the
      store, the detector and the input. c = 1 (round; any c << g works).
  The four wave neurons keep their rows (comparator level, rotation B, the margin's deliveries from C1 and C7) except
      the start: they no longer read it from the detector at the record's next step; they read B s0 (H7 - C7) / c, so
      the start enters their rotation one step later, from H7. Every reader of the wave (the linear route's forward
      copy, the readout rows, the wave neurons themselves) that took u . z from them now takes u . (their wave) +
      (u . s0 / c) (H7 - C7). Then u . (their wave) + (u . s0)(H7 - C7)/c = u . z_t of w5b_slot7 at every step: the same
      function while nothing clips differently.
  So the start passes through ARC's hiding slot (silence H7 one step after a record and the wave never starts), and the
  rotation that carries the wave from then on lives in the four former wave neurons, which the map now puts in a memory
  slot.
MAP (w6b_map): slot 7 = H7 (RMS scale against ARC's neuron 7). Slot 3 = the four former wave neurons, equal weights,
  RMS scale of their sum: ARC's neuron 3 is the memory neuron that reads the hiding neurons most strongly (+1.20 h1,
  -1.07 h8; neuron3/README.md), and these neurons are the memory part that takes the hiding slot's start. The delay
  cell that held slot 3 (mem6, position 1 at the end) becomes a helper in no slot. Other rows as w5b_slot7.
Extra neurons over 16 (6): w5b_slot7's five (late gate, C7, C1, C6, CR) and H7.
"""
import numpy as np
from algzoo.handbuilt_w5b import build_w5b, w5b_map, WAVE


def build_w6b(c=1.0, a=1000.0, g=1e4, **kw):
    wi0, wh0, wo0, idx = build_w5b(**kw)
    n0 = len(wi0); n = n0 + 1; H = n0; idx = dict(idx); idx["h7"] = H
    wi = np.zeros(n); wh = np.zeros((n, n)); wo = np.zeros((10, n))
    wi[:n0] = wi0; wh[:n0, :n0] = wh0; wo[:, :n0] = wo0
    Wn = [idx[k] for k in WAVE]; C7, D = idx["c7"], idx["detect"]
    # the start per unit of the jump: the wave neurons' detector weights minus the comparator level's (g / a)
    s0 = (wh[Wn, D] - g / a) * a
    # H7 = C7's row plus c on the jump
    wi[H] = wi[C7]; wh[H] = wh[C7]; wh[H, D] += c / a
    # every reader of the wave (computed before the wave neurons lose their start): u . s0 / c on (H7 - C7)
    R = wh[:, Wn] @ s0; Ro = wo[:, Wn] @ s0
    wh[:, H] += R / c; wh[:, C7] -= R / c
    wo[:, H] += Ro / c; wo[:, C7] -= Ro / c
    # the wave neurons no longer take the start from the detector
    wh[Wn, D] -= s0 / a
    return wi, wh, wo, idx


def w6b_map(wi, wh, wo, idx, n_lists=20000, seed=0):
    """w5b's map with slot 7 = H7 and slot 3 = the four former wave neurons (mem6 a helper)."""
    import torch as th
    from algzoo.core import example
    H = idx["h7"]; Wn = [idx[k] for k in WAVE]
    n0 = H  # the first n0 neurons are w5b's
    M5, lay = w5b_map(wi[:n0], wh[:n0, :n0], wo[:, :n0], {k: v for k, v in idx.items() if v < n0}, n_lists=n_lists, seed=seed)
    M = np.zeros((16, len(wi))); M[:, :n0] = M5; M[7] = 0; M[3] = 0
    x = th.randn(n_lists, 10, generator=th.Generator().manual_seed(seed), dtype=th.float64).numpy()
    def acts(Wi, Wh):
        h = np.zeros((len(x), len(Wi))); out = []
        for t in range(10):
            h = np.maximum(h @ Wh.T + x[:, t:t + 1] * Wi, 0); out.append(h)
        return np.stack(out[1:], 1).reshape(-1, len(Wi))
    A = [np.asarray(p, float) for p in example()]
    Aa = acts(A[0], A[1]); Ah = acts(wi, wh)
    rms = lambda v: np.sqrt((v ** 2).mean())
    M[7, H] = rms(Aa[:, 7]) / rms(Ah[:, H])
    M[3, Wn] = rms(Aa[:, 3]) / rms(Ah[:, Wn].sum(1))
    lay = dict(lay); lay[7] = "h7"; lay[3] = "+".join(WAVE)
    return M, lay

"""Hand-built v6a: r5a's network (handbuilt_v4a rules, handbuilt_v5a layout) with two changes.
Research notes: research/20261017_handbuilt/faithful_twin/LEDGER.md, branch r6a.

1. Memory ring tap (ring_tap): the ring loops delay cell mem<tap> back into the linear route with weight ring_w
   (v4a used the last cell, mem6). With tap 5 the ring has 7 neurons (linear route, mem0..mem5); its modes have modulus
   (|w| rho^5)^(1/7), so the 0.7-modulus rule gives |w| = 0.7^7 / 0.99^5 = 0.0866, rounded up to 0.087 (r4a's rule, 0.062 for tap 6:
   exactly 0.7 falls below the scorer's bar by rounding). The looped shared level is removed
   through the store (U rho^tap m), and the readout adds back the current largest's backward wave that the ring carries
   from slot p - n_ring into slot p: w rho^tap c A^(p - 9 - n_ring) z_9 (v4a's ring_comp with n_ring = tap + 2).
2. Gap in neuron 3 (gap_lam): the hiding slots hold the gap as a linear read,
       level/gc - Q/gq = (m_{t-1} - x_t) - relu(s_{t-1} - x_t) = gap_t   (no record at t; both clip to 0 at a record).
   Delay cell mem6 (ARC's slot 3, position 1's cell, read only by the readout once the ring tap is moved) adds
   lam x (that read + detector / a), which is the gap at every step (at a record the detector carries it).
   In score units this is gap_eff = lam rho^-6 (the readout reads mem6 with weight -rho^-6).
   The readout row of position 1 takes off gap_eff x the hiding read of the final state (without the detector).
   That equals gap_8 when x_9 is below the second; otherwise position 1 is lowered: by gap_eff (x_9 - s_8) when
   x_9 lies between the second and the largest (harmless, x_9 is the second), by gap_eff gap_8 after a record at
   step 9 (harmful only when the old largest is at position 1).
3. Last-step compensation (last_comp, tap 5 only): at step 9 the ring carries position 1's cell (mem5 at step 8) into
   the linear route, which is position 8's score. mem6 at step 9 holds the same cell one step on, so the readout row
   of position 8 takes it off: (w / rho) (mem6 - U rho^6 m_8 - lam x the hiding read of the final state).
"""
import numpy as np
from algzoo.handbuilt_v4a import build_v4a, wave_2d
from algzoo.handbuilt_v5a import to_arc  # noqa: F401  (re-exported for the candidates)


def build_v6a(ring_tap=5, ring_w=None, gap_eff=0.0, rho=0.99, U=1.0, s=20.0, gc=1e10, gq=1e4, kappa=0.5, a=1000.0,
              last_comp=False, **kw):
    net = build_v4a(bw_late=True, ring_w=0.0, ring_comp=False, rho=rho, U=U, s=s, gc=gc, gq=gq, kappa=kappa, a=a,
                    **kw)
    wi, wh, wo = net.params
    S, L, Qn = net.idx["store"], net.idx["linear"], net.idx["runnerup"]
    C = [net.idx[f"clock{k}"] for k in range(3)]
    Y = [net.idx[f"mem{j}"] for j in range(7)]
    from algzoo.handbuilt import TO_PQ
    Aw, bw, cw = wave_2d(K=1 + kappa)
    Ai = np.linalg.inv(Aw)
    if ring_w is None:   # smallest weight in steps of 0.001 whose modes reach modulus 0.7 (r4a's 0.062 for tap 6)
        ring_w = -np.ceil(1000 * 0.7 ** (ring_tap + 2) / rho ** ring_tap) / 1000
    if ring_w:
        n_ring = ring_tap + 2
        wh[L, Y[ring_tap]] += ring_w
        wh[L, S] += -ring_w * U * rho ** ring_tap / s
        for p in range(1, 8 if (last_comp and ring_tap == 5) else 9):   # last_comp takes slot 8's whole leak off
            u = ring_w * rho ** ring_tap * (cw @ np.linalg.matrix_power(Ai, 9 + n_ring - p))
            wo[p, C] += u @ TO_PQ
    if gap_eff:
        lam = gap_eff * rho ** 6
        for k in C:
            wh[Y[6], k] += lam / (3 * gc)
            wo[1, k] += gap_eff / (3 * gc)
        wh[Y[6], Qn] += -lam / gq
        wh[Y[6], net.idx["detect"]] += lam / a      # the detector carries the gap at a record: mem6 gets gap_8 exactly
        wo[1, Qn] += -gap_eff / gq                  # the readout takes off the hiding read only (no detector term)
    if last_comp and ring_w and ring_tap == 5:
        # at the last step the ring adds w x mem5(8) (position 1's cell, level removed) into the linear route, i.e.
        # into position 8's score. mem6(9) = rho mem5(8) + level bookkeeping + the gap term, so the readout row of
        # position 8 takes off (w / rho) (mem6 - U rho^6 m_8 - lam x hiding read of the final state).
        lam = gap_eff * rho ** 6
        wo[8, Y[6]] += -ring_w / rho
        wo[8, S] += ring_w / rho * U * rho ** 6 / s
        if gap_eff:
            for k in C:
                wo[8, k] += ring_w / rho * lam / (3 * gc)
            wo[8, Qn] += -ring_w / rho * lam / gq
    return net

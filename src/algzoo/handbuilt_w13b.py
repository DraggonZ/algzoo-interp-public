"""Hand-built w13b (wider track, 24 neurons): w12b_keepfirst's network (handbuilt_w12b on handbuilt_w8b) with a hiding
wave that has no backward part (w10b's design a4, which w12b's freed neuron makes fit), research/20261017_handbuilt/
faithful_twin/LEDGER.md, branch w13b.

Rules (build_w13b), all solved from w12b's rows:
- Old clocks' hide amount. With W the old clocks' read of the hiding comparators (H7, C1, C6), sp1 and sp2 the
  comparators' special parts one and two steps after a record (handbuilt_w8b.special_parts), R the old clocks' turn and
  r_p the readout's read of them at position p, the hide at the largest's own position is
      phi1 W sp1 + phi2 W sp2,   phi1 = r_p R^(7-p), phi2 = r_p R^(6-p)   (the same for every p: f(0) = 1).
- J (the freed linear-route neuron) carries minus that, three steps after a record: its row reads (H7, C1, C6) with
  weights solved like w8b's rebuild_reads: level-free (sum 0), zero one step after a record, and equal to the hide
  amount two steps after (so unclipped J is non-zero at age 3 only).
- Six clocks, one per wave coordinate: pairs turning 45, 90, 135 degrees (clock0/clock1, clock2/turn0, turn1/turn2).
  Each reads the comparator level as the old clocks did, removes it through C7 (clock_i(n+1) = level + sum_j R6_ij
  (clock_j(n) - C7(n)) + v_i J(n)), and takes J as its start (v = 1 on the cosine coordinate of each pair).
- Readout: positions 1-5 read the clocks (minus C7) with weights solved so that a record at p = 0..5 hides position p by
  J's value and no other position (6 equations, 6 unknowns); position 6 reads -J; position 7 reads minus J's read of the
  comparators (a record two steps before the end); positions 0, 8, 9 as w12b.
- Removed: the detector's backward write into mem3 (set to mem4's shared-level weight), the readout's matching read of
  H7 and C7 at position 4 (option), and the newest cell's forward copy of the old wave.
"""
import numpy as np
from algzoo.handbuilt_w12b import build_w12b, CLOCKS
from algzoo.handbuilt_w8b import special_parts


def rot(deg):
    t = np.radians(deg); return np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])


ANGLES = (45.0, 90.0, 135.0)


def build_w13b(drop_pos4="all", angles=ANGLES, report=False):
    wi, wh, wo, idx, _ = build_w12b()
    wi, wh, wo = wi.copy(), wh.copy(), wo.copy()
    C = [idx[k] for k in CLOCKS]            # clock0, clock1, clock2, turn0, turn1, turn2
    comps = [idx[k] for k in ("h7", "c1", "c6")]
    C7, J, D = idx["c7"], idx["linear"], idx["detect"]
    # --- old clocks' hide amount at the largest's position, per unit jump
    R = wh[np.ix_(C, C)]; W = wh[np.ix_(C, comps)]
    sp = special_parts(wi, wh, idx)
    sp1 = np.array([sp[k][0] for k in ("h7", "c1", "c6")]); sp2 = np.array([sp[k][1] for k in ("h7", "c1", "c6")])
    mp = np.linalg.matrix_power
    phi1 = np.array([wo[p, C] @ mp(R, 7 - p) for p in range(1, 8)])
    phi2 = np.array([wo[p, C] @ mp(R, 6 - p) for p in range(1, 7)])
    assert np.allclose(phi1 @ W @ sp1, (phi1 @ W @ sp1)[0]) and np.allclose(phi2 @ W @ sp2, (phi2 @ W @ sp2)[0])
    hide = float(phi1[0] @ W @ sp1 + phi2[0] @ W @ sp2)      # about -1.5 (w8b: K = 1.5)
    # --- J's row: level-free, 0 at age 1, -hide at age 2 (reads at age 2, holds it at age 3)
    A = np.array([np.ones(3), sp1, sp2]); x = np.linalg.solve(A, np.array([0.0, 0.0, -hide]))
    # J is a comparator: C7's level plus the hide part, so that a record clips it (a pending start dies with the record);
    # its readers read J - C7
    wh[J, :] = wh[C7]; wi[J] = wi[C7]; wh[J, comps] += x
    # --- the newest cell's forward copy of the old wave, and all old reads of the clocks, go
    for r in range(len(wi)):
        if r not in C: wh[r, C] = 0.0
    wo[:, C] = 0.0
    # --- backward write into mem3 removed
    wh[idx["mem3"], D] = wh[idx["mem4"], D]
    if drop_pos4:
        wo[4, idx["h7"]] = 0.0; wo[4, idx["c7"]] = 0.0
    if drop_pos4 == "all":   # also the late gate's, detector's and canceller's reads at position 4 (same backward write)
        wo[4, idx["late"]] = 0.0; wo[4, D] = 0.0; wo[4, idx["cancel"]] = 0.0
    # the old wave's margin reads at positions 3 and 7 (a C1 - C7 pair: position 7's margin, position 3's backward
    # spill of it) go; position 7's whole hide is read below
    c1 = idx["c1"]
    for r in (3, 7):
        wo[r, C7] += wo[r, c1]; wo[r, c1] = 0.0
    # --- new clocks: level as before (wi, store, detect), turn by R6 around C7's level, start from J
    lev = wh[C[0]].copy(); lev[C] = 0.0; lev[comps] = 0.0
    R6 = np.zeros((6, 6))
    for k, a in enumerate(angles): R6[2 * k:2 * k + 2, 2 * k:2 * k + 2] = rot(a)
    v = np.array([1.0, 0, 1.0, 0, 1.0, 0])
    for i, c in enumerate(C):
        wh[c] = lev; wi[c] = wi[C[0]]
        wh[c, C] = R6[i]; wh[c, C7] = -R6[i].sum() - v[i]; wh[c, J] = v[i]
    # --- readout: positions 1..5 from the clocks; a record at p = 0..5 leaves R6^(5-p) v J at the end
    Sm = np.stack([mp(R6, 5 - p) @ v for p in range(6)], 1)          # columns: records at p = 0..5
    T = np.zeros((6, 6)); T[np.arange(1, 6), np.arange(1, 6)] = -1.0     # rows: positions 0..5
    Rr = T @ np.linalg.inv(Sm)                                          # Rr[j] . s_p = T[j, p]
    for j in range(1, 6):
        wo[j, C] = Rr[j]; wo[j, C7] -= Rr[j].sum()
    wo[6, J] = -1.0; wo[6, C7] += 1.0
    wo[7, comps] = -x
    info = dict(hide=hide, J_row=x, cond=np.linalg.cond(Sm), readout=Rr[1:6])
    if report: print(info)
    return wi, wh, wo, idx, info

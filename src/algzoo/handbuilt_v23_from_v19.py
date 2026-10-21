"""Hand-built v23 from v19: 16 neurons on v19's job assignment, with a ring memory, a negative imprint of the largest
in memory, and votes that enter by two routes.

Kept from v19 unchanged (rules in handbuilt_v19.py): the store S and detector D (largest so far m1), the clock C0-C2
(level g (th m1 - x) plus a 2-D phase turning 45 degrees per step, wiped by a record, restarted one step later with
K d), the flip F (same level and wipe, phase part flips sign each step, receives beta K d), the keep gate G (one step
late, the clipped vote V = A0 (x - K jump) - eps m1 plus a level), the flag Z (relu(previous jump + w (x - m1)), on
only when two records arrive in a row), and the readout of the clock, flip and flag. K = 1 + kappa: a record shows
A0 (p - kappa d) and its wave carries K d (v19's kappa). Position 9 reads V directly, minus 1000 D.

Changed, with the rule for each:

1. Negative imprint (brief point 1). Every memory slot holds its score content subtracted from an imprint U m1 of the
   largest so far: neuron = U m1 - (vote + wave). The readout reads -1 x the slot plus U m1 from the store. Each jump
   of the largest adds U d to every slot (the detector's broadcast); a new slot gets its imprint from the store, so
   the store "discounts later numbers by the largest". U = A0 (1 + 2K): the smallest imprint that keeps every slot
   positive in the worst case (a vote of m1, plus the current record's wave peak K A0 d <= K A0 m1, plus a revealed
   record's spill of the same size).

2. Two routes (brief point 3). A share r1 = 1/2 of each vote is written one step after the number arrives, into the
   newest slot, from the clock's level (whose input weight -g carries the number: x = th m1 - level / g), the store
   and the detector; the other half one step later from the keep gate. Both carry the same clipped vote V, so a
   record writes A0 (p - kappa d) by both routes. The keep gate's level gate_level x A0 m1 keeps it linear for every
   number above -gate_level x m1, so the two routes agree. Position 9's vote has only the direct route (read at the
   end from the clock's level and the detector). Position 8's vote is read from the keep gate at the end (full size).

3. Ring memory (brief point 4). Delay frame: slot j holds number t-1-j at step t (positions 8..1 at the end); slot 7
   feeds slot 0 with weight ep (the loop gain), so position 0 comes round into slot 0 at the last step, scaled by ep,
   and is read from there (gain 1/ep, after removing number 8's direct share, which the keep gate also holds). The
   ring frame (default) is the exact rescaling slot_j -> rho^j slot_j with rho = ep^(1/8): every slot feeds the next
   with weight rho, a uniformly fading rotating shift register whose modes turn by 0, 45, 90, 135 and 180 degrees per
   step and shrink by rho. Rules that keep the bookkeeping exact for the record that is still the largest:
   - the newest slot takes (1 - ep) U m1 from the store, because the wrap brings ep U m1;
   - a record's backward half goes into slots 1-7 as in v19 (one slot later, since slots are one step fresher), and
     ep x the wave's peak into slot 0 (the record's own slot now, and position 0 when the record is at step 8);
   - the forward copies of the clock's wave (into slot 1, one step after the slot's creation, when the clock has
     restarted) and the flag's missing peak carry (1 - ep) of v19's, because the record's backward half written into
     not-yet-used slots comes back through the wrap with weight ep, at exactly the forward wave (8-periodic).
   What cannot be made exact: an overtaken record leaves ep x its wave in the slots written after the overtaking (the
   wall in bridge/README.md). The clock knows only the current record, and the not-yet-used slots are the only place
   that holds the old one, so nothing outside the ring can cancel it (README of superposition/v23/from_v19).
   ep = 1e-4 by rule: position 0's readout gain 1/ep is then no larger than v19's largest gains (g, w = 1e4), and the
   leftover is below v19's own tie-breaker. bwd=False removes all backward writes ("forward-only hiding"): the ring is
   then exact at any loop gain, but the clock's wave is uncompensated before the largest.
"""
import numpy as np
from algzoo.handbuilt import Net, FROM_PQ, TO_PQ, _rot

NM = 8
TURN = 2 * np.pi / NM
TH = TURN * np.arange(NM)


def build_v23f(beta=0.5, a=1000.0, s=20.0, g=1e4, c=0.75, U=None, w=1e4, eps=1e-4, kappa=0.5, delta=0.0,
               ep=1e-4, r1=0.5, gate_level=4.0, ring=True, bwd=True):
    if ep <= 0:
        raise ValueError("position 0 is held only by the wrap (ep > 0)")
    K = 1 + kappa                   # the wave carries K x the jump
    A0 = 1 - c                      # vote per unit of value; also the wave's peak per unit of jump
    A = A0 / (1 + beta)             # share of the peak carried by the 45-degree part; the flip part carries beta * A
    U = A0 * (1 + 2 * K) if U is None else U   # imprint per unit of m1 (positivity rule above)
    r2 = 1 - r1                     # the keep gate's share of each vote
    net = Net(d=16)
    S, D, G, F = (net.add(k) for k in ("store", "detect", "keep", "flip"))
    C = [net.add(f"clock{k}") for k in range(3)]
    Z = net.add("flag")
    Y = [net.add(f"mem{j}") for j in range(NM)]
    wi, wh, wo = net.params
    m1 = {S: 1 / s, D: 1 / a}                         # largest so far, read from the state
    lev = {k: 1 / 3 for k in C}                       # clock level (the three neurons' mean)
    th = 1 - delta
    xprev = {**{j: th * v for j, v in m1.items()}, **{k: -v / g for k, v in lev.items()}}   # last input
    R = _rot(TURN)

    # largest so far (v19)
    wh[S, S] = 1
    wh[S, D] = s / a
    wi[D] = a
    wh[D, D] = -1
    wh[D, S] = -a / s

    # clock (v19)
    wh[np.ix_(C, C)] = FROM_PQ @ R @ TO_PQ
    for j, v in m1.items():
        wh[C, j] += g * th * v
    wi[C] = -g
    wh[C, D] += K * FROM_PQ @ (R.T @ np.array([1.0, 0.0])) / a

    # flip (v19)
    wi[F] = -g
    wh[F, S] = g * th / s
    wh[F, D] = (g * th + K * beta) / a
    wh[F, F] = -1.0
    for k, v in lev.items():
        wh[F, k] += v
    flip = {F: 1.0, **{k: -v for k, v in lev.items()}}

    # the clipped vote of the last number as a linear function of the state (v19's V)
    V = {j: A0 * v for j, v in xprev.items()}
    V[D] = V.get(D, 0) - K * A0 / a
    V[S] = V.get(S, 0) - eps / s
    # keep gate: V plus a level that keeps it linear
    LG = gate_level * A0
    for j, v in V.items():
        wh[G, j] += v
    for j, v in m1.items():
        wh[G, j] += LG * v

    # flag (v19)
    wi[Z] = w
    wh[Z, S] = -w / s
    wh[Z, D] = (1 - w) / a

    # memory, delay frame: slot j holds number t-1-j at step t; slot 7 wraps into slot 0 with weight ep.
    # Neuron = U m1 - score content, so every score-content write enters with a minus sign.
    for j in range(1, NM):
        wh[Y[j], Y[j - 1]] = 1.0
    wh[Y[0], Y[NM - 1]] = ep
    # imprint: the newest slot gets (1 - ep) U m1 from the store (the wrap brings ep U m1); every slot U x each jump
    wh[Y[0], S] += (1 - ep) * U / s
    for y in Y:
        wh[y, D] += U / a
    # backward half of a record's wave: v19's slot k becomes slot k + 1; slot 0 gets ep x the peak (offset -8 = 0)
    fwd = (1 - ep) if bwd else 1.0
    if bwd:
        for k in range(NM - 1):
            wh[Y[k + 1], D] -= K * (A * np.cos(TH[k] + TURN) + A * beta * (-1) ** (k + 1)) / a
        wh[Y[0], D] -= ep * K * A0 / a
    # route 1: the direct share of the last number, into the newest slot
    for j, v in V.items():
        wh[Y[0], j] -= r1 * v
    # route 2: the keep gate's share of the number before, minus its level (the store then holds the same m1)
    wh[Y[1], G] -= r2
    wh[Y[1], S] += r2 * LG / s
    # forward copies of the clock-plus-flip wave for the number before (slot 1), fwd x v19's
    wh[Y[1], C] -= fwd * A * np.array([np.cos(-TURN), np.sin(-TURN)]) @ TO_PQ
    for j, v in flip.items():
        wh[Y[1], j] -= fwd * A * v
    # flag: the first record's missing peak when two records arrive in a row, fwd x v19's
    wh[Y[1], Z] -= fwd * K * A0
    wh[Y[1], D] += fwd * K * A0 * w / a

    # readout. Positions 1-7 read slot 8-p; position 0 reads slot 0 / ep after removing number 8's direct share;
    # position 8 reads the keep gate; every memory position subtracts the clock-plus-flip wave as in v19.
    for p in range(1, NM):
        wo[p, Y[NM - p]] += -1.0
        wo[p, S] += U / s
    wo[0, Y[0]] += -1.0 / ep
    wo[0, S] += U / (ep * s)
    wo[0, G] += -r1 / ep
    wo[0, S] += r1 * LG / (ep * s)
    for p in range(NM):
        jp = 7 - p
        wo[p, C] += -A * np.array([np.cos(TH[jp]), np.sin(TH[jp])]) @ TO_PQ
        for j, v in flip.items():
            wo[p, j] += -A * (-1) ** p * v
    wo[8, G] += 1.0
    wo[8, S] += -LG / s
    wo[8, Z] += K * A0
    wo[8, D] += -K * A0 * w / a
    if bwd:  # a record at 8 writes its wave's far end into position 0; removed if a record at 9 overtakes it
        wo[0, Z] += -K * A0
        wo[0, D] += K * A0 * w / a
    for j, v in V.items():
        wo[9, j] += v
    wo[9, D] += -1000.0

    if ring:  # exact rescaling slot_j -> rho^j slot_j: every ring weight becomes rho
        rho = ep ** (1 / NM)
        sc = np.ones(16)
        sc[Y] = rho ** np.arange(NM)
        rescale(net, sc)
    return net


def rescale(net, sc):
    """Exact symmetry of a bias-free ReLU network: neuron i -> sc[i] x neuron i (sc > 0)."""
    wi, wh, wo = net.params
    wi *= sc
    wh *= sc[:, None] / sc[None, :]
    wo /= sc[None, :]


def scores(net, X):
    wi, wh, wo = net.params
    h = np.zeros((len(X), len(wi)))
    for t in range(X.shape[1]):
        h = np.maximum(h @ wh.T + X[:, t:t + 1] * wi, 0)
    return h @ wo.T

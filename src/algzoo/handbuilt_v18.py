"""Hand-built v18: 16 neurons, the trained network's layout with two roles reassigned. About 96% accurate.

Every weight below follows from the rule written next to it
nothing is fitted. Designed on top of v17
scripts/handbuilt_v18.py there wraps this module. The parts, with the trained neurons they copy:

  store S, detector D (trained 2, 4): the largest so far. At step t, S = s * m1(t-1) and D = a * (jump at t), so
      m1(t) = S/s + D/a. D fires only at a new largest (a "record").
  clock C0-C2 (trained 1, 6, 7): a level g * (m1(t-1) - x(t)) plus a 2-D phase that turns 45 degrees per step. A record
      wipes it (input weight -g)
      one step later it receives the record's jump d from the detector. Its level also
      records the last input, so x(t-1) = m1(t-1) - level / g is available one step later ("xprev").
  flip F (replaces the trained late gate 9): a fourth clock neuron, same level and wipe, whose phase part flips sign each
      step and receives beta * d. Clock plus flip hold the wave d (cos(45 k) + beta (-1)^k) / (1 + beta) around the
      record, k = offset from it. The flip makes the revealed wave sharper: first neighbours fall from 0.71 to
      (0.71 - beta) / (1 + beta) of the peak.
  keep G (trained 5): one step late, writes (1 - c) x for a positive number and (1 - c) p for a record (p = the
      previous largest), minus a tiny eps * m1 that breaks the tie between the current largest's position and the
      previous largest's position. No forgetting: with exact memory levels nothing needs it.
  flag Z (replaces the trained second-largest store 8): relu(previous jump + w (x - m1)). Equals the previous record's
      jump (plus w x the new jump, cancelled downstream) only when two records arrive in a row
      otherwise zero.
  memory Y0-Y7 (trained 0, 3, 10-15): a delay line. Slot j holds position t-2-j at step t. Each slot carries the level
      lam * m1 exactly (the newest slot from the store, older slots by adding each jump), so nothing is subtracted.
      The newest slot gets the keep gate's vote and copies the clock-plus-flip wave value at its position (forward
      copies)
      a record's detector pulse writes the backward half of its wave into the older slots. The flag writes
      the first record's missing peak when two records arrive in a row (the second one wipes the clock before the
      clock could copy that peak).
  readout: positions 0-7 read their slot minus lam * m1 minus the clock-plus-flip wave (hiding the current largest);
      position 8 reads the keep gate (and the flag, for records at steps 8 and 9)
      position 9 reads the keep gate's
      formula directly, minus 1000 * D so that a record at step 9 is not named.

What follows for scores: a lone record leaves all scores at zero (hidden exactly)
an overtaken record shows
(1 - c) x its value at its position, with a sharpened wave around it that reaches only positions written before the
new record arrived; kept numbers show (1 - c) x their value."""
import numpy as np
from algzoo.handbuilt import Net, FROM_PQ, TO_PQ, _rot

NM = 8
TURN = 2 * np.pi / NM
TH = TURN * np.arange(NM)


def build_v18(beta=0.5, a=1000.0, s=20.0, g=1e4, c=0.75, lam=10.0, w=1e4, eps=1e-4):
    A0 = 1 - c                      # vote per unit of value; also the wave's peak per unit of jump
    A = A0 / (1 + beta)             # share of the peak carried by the 45-degree part; the flip part carries beta * A
    net = Net(d=16)
    S, D, G, F = (net.add(k) for k in ("store", "detect", "keep", "flip"))
    C = [net.add(f"clock{k}") for k in range(3)]
    Z = net.add("flag")
    Y = [net.add(f"mem{j}") for j in range(NM)]
    wi, wh, wo = net.params
    m1 = {S: 1 / s, D: 1 / a}                         # largest so far, read from the state
    lev = {k: 1 / 3 for k in C}                       # clock level (the three neurons' mean)
    xprev = {**m1, **{k: -v / g for k, v in lev.items()}}   # last input = m1 - level / g
    R = _rot(TURN)

    # largest so far
    wh[S, S] = 1
    wh[S, D] = s / a
    wi[D] = a
    wh[D, D] = -1
    wh[D, S] = -a / s

    # clock: level g m1 - g x, phase turns 45 degrees, wiped by a record, gets the jump one step later at the angle a
    # two-step-old number has one step early
    wh[np.ix_(C, C)] = FROM_PQ @ R @ TO_PQ
    for j, v in m1.items():
        wh[C, j] += g * v
    wi[C] = -g
    wh[C, D] += FROM_PQ @ (R.T @ np.array([1.0, 0.0])) / a

    # flip: same level and wipe; its phase part a -> -a each step, plus beta x the jump
    wi[F] = -g
    wh[F, S] = g / s
    wh[F, D] = (g + beta) / a
    wh[F, F] = -1.0
    for k, v in lev.items():  # restores the level that the self-weight -1 removed
        wh[F, k] += v
    flip = {F: 1.0, **{k: -v for k, v in lev.items()}}    # flip phase part = F - clock level

    # keep gate: (1 - c) x last input, minus (1 - c) jump (so a record writes (1 - c) p), minus the tiny tie-breaker
    V = {j: A0 * v for j, v in xprev.items()}
    V[D] = V.get(D, 0) - A0 / a
    V[S] = V.get(S, 0) - eps / s
    for j, v in V.items():
        wh[G, j] += v

    # flag: relu(w (x - m1) + previous jump)
    wi[Z] = w
    wh[Z, S] = -w / s
    wh[Z, D] = (1 - w) / a

    # memory: delay line with exact level
    for j in range(1, NM):
        wh[Y[j], Y[j - 1]] = 1.0
    for j, v in m1.items():
        wh[Y[0], j] += lam * v
    for k, y in enumerate(Y):
        if k > 0:  # level follows each jump
            wh[y, D] += lam / a
        wh[y, D] += (A * np.cos(TH[k] + TURN) + A * beta * (-1) ** (k + 1)) / a   # backward half of a record's wave
    wh[Y[0], G] += 1.0                                                    # the vote
    wh[Y[0], C] += A * np.array([np.cos(-TURN), np.sin(-TURN)]) @ TO_PQ   # forward copy of the 45-degree part
    for j, v in flip.items():  # forward copy of the flip part
        wh[Y[0], j] += A * v
    wh[Y[0], Z] += A0  # missing peak when two records are in a row
    wh[Y[0], D] += -A0 * w / a

    # readout
    for p in range(NM):
        jp = 7 - p
        wo[p, Y[jp]] += 1.0
        wo[p, S] += -lam / s
        wo[p, C] += -A * np.array([np.cos(TH[jp]), np.sin(TH[jp])]) @ TO_PQ
        for j, v in flip.items():
            wo[p, j] += -A * (-1) ** p * v
    wo[8, G] += 1.0
    wo[8, Z] += A0  # record at 8 overtaken at 9: its peak
    wo[8, D] += -A0 * w / a
    # a record at 8 writes its wave's far end into the slot that ends as position 0 (offset -8 wraps to 0); if a record
    # at 9 then overtakes it, the clock never hides that, so the flag (on only then) removes it
    wo[0, Z] += -A0
    wo[0, D] += A0 * w / a
    for j, v in V.items():
        wo[9, j] += v
    wo[9, D] += -1000.0
    return net


def scores(net, X):
    wi, wh, wo = net.params
    h = np.zeros((len(X), len(wi)))
    for t in range(X.shape[1]):
        h = np.maximum(h @ wh.T + X[:, t:t + 1] * wi, 0)
    return h @ wo.T



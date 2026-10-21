"""Hand-built v23: ARC's corrected mechanism, built from scratch in 16 neurons. Every weight follows from a rule below.

Research notes: research/20261017_handbuilt/superposition/v23/README.md.

Scores in this network, "relative" meaning before a level that every position shares is removed by the readout:

    every number x:       alpha x + beta relu(x)   (A0 x for a positive number, A0 = alpha + beta)
    the current largest:  A0 p - A0 kappa d        (p the previous largest, d its jump over p)

The memory holds each vote next to -alpha m1(at writing) + the shared level; the shared level grows with every jump, so
at the end every slot holds the same (LF - alpha) m1, which the readout removes. So the current largest is hidden
down to the previous largest's vote minus a margin, and an overtaken largest is revealed.

Neurons and the points of the account they implement (numbers refer to the design brief, an internal working note, not included):

  store S, detector D: the largest so far, as in ARC (S = s m1(t-1), D = a relu(x - m1(t-1))).
  clock C0-C2 and flip P (the four hiding neurons): each holds g (m1(t-1) - x_t), a comparator that a new largest wipes,
      plus a wave around the current largest's position: a 45-degree rotating pair (clock) and a sign that flips each
      step (flip), started one step after a new largest by the detector with size K d, K = 1 + kappa. The readout
      subtracts the wave at every position.
  linear route L (point 3, the direct route): one step after x_t arrives, L = alpha (x_t - m1) + LF m1, read straight off
      the clock's comparator level (x_t - m1 = -level / g). Its negative imprint of the largest, -alpha m1, comes from the
      store (point 1). L also carries the forward copy of the hiding wave at its position (point 2: the wave passes from
      the hiding neurons into the memory).
  keep gate G (point 3, the gated route): one step after x_t, G = beta relu(x_t). A number at the last step is read
      only through the direct route (the clock's level at the end).
  memory Y0-Y7 (point 4): an eight-slot shift that loops back (Y7 -> Y0 with weight w). Its modes are rotations by
      0, 45, 90, 135 and 180 degrees per step, of size w^(1/8). Slot j holds position t-2-j at step t; L and G write
      the newest slot. Within ten steps the loop only carries slots that no position has used yet, so w sets how much
      of what was written there comes back:
        - the shared level: every slot gets (LF - alpha) x each jump (the level follows the largest and the imprint is
          updated, point 1); the store removes the looped-back part of it from the newest slot, so the level is exact
          for every w;
        - a new largest's backward wave: the detector writes it into the older slots; the part that lands in unused
          slots comes back, after w, as its forward wave at the positions written later. The forward copies through L
          are scaled by 1 - w so that, while the largest stands, the two add up exactly. After it is overtaken the
          looped-back share w stays at positions after the overtaking: that is the price of looping back, and the
          experiment of this network.
  readout: positions 0-7 read their slot, position 8 reads L and G, position 9 reads the clock's level (the last input);
      every position subtracts the shared level and the hiding wave (position 9: no wave; it holds no copy).

w = 0 gives a one-way delay line with exact bookkeeping; w = 1 a full loop.
"""
import numpy as np
from algzoo.handbuilt import Net, FROM_PQ, TO_PQ, _rot

NM = 8
TURN = 2 * np.pi / NM


def wave_coef(shape="forward_safe", beta_f=0.5):
    """The hiding wave per unit of K d: f(k) = a cos(45 k) + b sin(45 k) + c (-1)^k with f(0) = a + c = 1 (k = offset from
    the largest). Four hiding neurons allow only these three parts: the clock's level is taken by its comparator.

    "symmetric" (v18, v19): b = 0, c = beta_f / (1 + beta_f).
    "forward_safe" (v23): when a largest is overtaken its wave is revealed. Positions after it hold numbers that may sit
    just below it, so a positive spill there names them; positions before it hold numbers at least its jump d below it,
    so a spill of up to 1/K there is safe. Rule: no spill on the next two positions, f(1) = f(2) = 0. That fixes
    c = 1 / (2 + sqrt 2) = 0.293, b = -c, a = 1 - c; the earlier positions then get f(-1) = 0.41 and f(-2) = 0.59, both
    below 1/K for K = 1.5."""
    if shape == "symmetric":
        c = beta_f / (1 + beta_f); return 1 - c, 0.0, c
    c = 1 / (2 + np.sqrt(2)); return 1 - c, -c, c


def wave(k, coef):
    a, b, c = coef
    return a * np.cos(TURN * k) + b * np.sin(TURN * k) + c * (-1.0) ** k


def build_v23(w=0.0, A0=1.0, share=0.5, kappa=0.5, shape="forward_safe", beta_f=0.5, a=1000.0, s=20.0, g=1e4, lf=None,
              far=True, sign=-1, U_mem=None, rho=1.0, level_exact=True):
    """share: the gated route's share of a vote (beta / A0); lf: the linear route's level per unit of m1 (rule below);
    far: write the backward wave's far end (offset -8) too. It matters only for a largest at step 8: with it, position 0
    is hidden correctly while that largest stands, but shows a full peak if step 9 overtakes it;
    sign: -1 (default) stores each slot as the shared level minus its vote and waves, so the memory's part of a score is
    a vote next to a larger negative imprint of the largest, which the store adds back at the readout (point 1, as
    ARC's steering tests found); +1 stores level plus vote, the same function, opposite under nudges of the memory;
    U_mem: the memory's shared level (imprint) per unit of m1, default lf - alpha;
    rho: the memory fades by rho per step (ARC's level mode: 0.99); every vote and wave is read back exactly by readout
    gains rho^-(age), and the level written when a slot is filled is exact too;
    level_exact: if False, each jump adds the same level to every slot (the store feeding the memory along one direction,
    as in ARC) instead of adding it pre-faded by the slot's age, so slots filled earlier end up with slightly more of the
    later jumps: the largest leaks into decisions by U (rho^-k - 1) per unit of a jump, k the steps between filling and
    jump."""
    beta = share * A0
    alpha = A0 - beta
    K = 1 + kappa
    coef = wave_coef(shape, beta_f)
    Ap = A0 * np.hypot(coef[0], coef[1])   # size of the rotating part per unit of K d
    phi = np.arctan2(coef[1], coef[0])     # its phase
    Af = A0 * coef[2]                      # size of the flip part
    if lf is None:                   # L stays positive for x >= -2 m1 with the deepest wave lobe and the looped-back peak
        lf = alpha + 2 * alpha + K * A0 * (1 + w) + K * A0
    U = lf - alpha if U_mem is None else U_mem      # shared memory level per unit of m1
    net = Net(d=16)
    S, D, G, L, P = (net.add(k) for k in ("store", "detect", "keep", "linear", "flip"))
    C = [net.add(f"clock{k}") for k in range(3)]
    Y = [net.add(f"mem{j}") for j in range(NM)]
    wi, wh, wo = net.params
    m1 = {S: 1 / s, D: 1 / a}                       # largest so far, read from the state
    lev = {k: 1 / 3 for k in C}                     # clock level = g (m1(t-1) - x_t)
    xprev = {**m1, **{k: -v / g for k, v in lev.items()}}   # the last input
    R = _rot(TURN)

    def add(row, terms, c=1.0, M=wh):
        for j, v in terms.items():
            M[row, j] += c * v

    # largest so far
    wh[S, S] = 1
    wh[S, D] = s / a
    wi[D] = a
    wh[D, D] = -1
    wh[D, S] = -a / s

    # clock: level g m1 - g x, 45-degree turn, started by the jump (angle -45 degrees one step after the record)
    wh[np.ix_(C, C)] = FROM_PQ @ R @ TO_PQ
    for k in C:
        add(k, m1, g)
    wi[C] = -g
    wh[C, D] += K * FROM_PQ @ (R.T @ np.array([1.0, 0.0])) / a
    # flip: same level and wipe; its own part changes sign each step and gets beta_f K d
    wi[P] = -g
    wh[P, S] = g / s
    wh[P, D] = (g + K * Af / Ap) / a           # the flip neuron's own part is read with weight Ap
    wh[P, P] = -1.0
    add(P, lev)                                     # restores the level the self-weight -1 removed
    flip = {P: 1.0, **{k: -v for k, v in lev.items()}}   # flip part = P - clock level

    # keep gate: beta relu(last input)
    add(G, xprev, beta)
    # linear route: alpha (x - m1) + lf m1 = -alpha level / g + lf m1; minus the looped-back peak w K A0 d;
    # plus (1 - w) x the forward copy of the wave at its own position (offset t - r: clock angle + 90 degrees, flip x -1)
    add(L, {k: -alpha / g * v for k, v in lev.items()})
    add(L, m1, lf)
    wh[L, D] += -w * K * A0 / a
    wh[L, C] += (1 - w) * Ap * np.array([np.cos(phi - 2 * TURN), np.sin(phi - 2 * TURN)]) @ TO_PQ
    add(L, flip, -(1 - w) * Ap)

    # memory: shift, loop back, writes
    for j in range(1, NM):
        wh[Y[j], Y[j - 1]] = rho
    wh[Y[0], Y[NM - 1]] = w
    wh[Y[0], L] += sign
    wh[Y[0], G] += sign
    add(Y[0], {S: 1 / s}, U - sign * (lf - alpha))  # the slot's level is U m1 whatever L's level and the sign
    add(Y[0], {S: 1 / s}, -w * U)                   # the looped-back share of the shared level (m1 two steps back)
    for j, y in enumerate(Y):
        wh[y, D] += U * (rho ** j if level_exact else 1.0) / a   # level follows each jump; imprint updated
        if far or j < NM - 1:
            wh[y, D] += sign * K * A0 * wave(-(j + 1), coef) * rho ** j / a   # backward wave (offsets -1 .. -8), pre-faded

    # readout
    def hide(p, c=1.0):                             # subtract c x the wave at position p
        th = TURN * (7 - p) + phi
        wo[p, C] += -c * Ap * np.array([np.cos(th), np.sin(th)]) @ TO_PQ
        add(p, flip, -c * Ap * (-1.0) ** p, wo)
    for p in range(NM):
        wo[p, Y[7 - p]] += sign * rho ** -(7 - p)
        add(p, {S: 1 / s}, -sign * U, wo)           # the shared level holds m1(8): the last jump never reaches the memory
        hide(p)
    wo[8, L] += 1.0
    wo[8, G] += 1.0
    add(8, {S: 1 / s}, -(lf - alpha), wo)            # L holds alpha x + (lf - alpha) m1(8)
    hide(8, 1 - w)                                  # its copy carries 1 - w of the wave; no looped-back share
    # position 9: A0 x9 - A0 K d9 (a record at the last step is hidden like any other)
    add(9, xprev, A0, wo)
    wo[9, D] += -A0 * K / a
    return net


def scores(net, X):
    wi, wh, wo = net.params
    h = np.zeros((len(X), len(wi)))
    for t in range(X.shape[1]):
        h = np.maximum(h @ wh.T + X[:, t:t + 1] * wi, 0)
    return h @ wo.T


def spread_memory(net, lam, seed=0, s=20.0, a=1000.0):
    """The same network with its memory stored spread across neurons (an exact change of basis while nothing switches off).

    New memory neurons z = Q y + lam m1(t-1) (1, ..., 1), with Q = (shared part) + (a seeded random rotation of the part of
    the slots that sums to zero). The shared part keeps v23's level in every neuron; lam adds more. m1(t-1) is the store
    over s. Every weight into, within and out of the memory is rewritten so that, with every neuron on, the function is
    unchanged; with a low lam neurons switch off on their mixed contents, as ARC's memory neurons do."""
    import copy
    net = copy.deepcopy(net)
    wi, wh, wo = net.params
    Y = [net.idx[f"mem{j}"] for j in range(NM)]
    S, D = net.idx["store"], net.idx["detect"]
    O = [k for k in range(len(wi)) if k not in Y]
    one = np.ones(NM)
    P1 = np.outer(one, one) / NM
    rng = np.random.default_rng(seed)
    B = np.linalg.qr(np.c_[one, rng.standard_normal((NM, NM - 1))])[0][:, 1:]   # basis of the zero-sum part
    Rr = np.linalg.qr(rng.standard_normal((NM - 1, NM - 1)))[0]
    Q = P1 + B @ Rr @ B.T
    Wyy, Wyo, Wout = wh[np.ix_(Y, Y)].copy(), wh[np.ix_(Y, O)].copy(), wo[:, Y].copy()
    A = Q @ Wyy @ Q.T
    wh[np.ix_(Y, Y)] = A
    wh[np.ix_(Y, O)] = Q @ Wyo
    # level: - lam A 1 m1(t-1) carried over, + lam m1(t) = lam (S/s + D/a) added
    wh[Y, S] += -lam * (A @ one) / s + lam / s
    wh[Y, D] += lam / a
    wo[:, Y] = Wout @ Q.T
    wo[:, S] += -lam * (Wout @ Q.T @ one) / s
    return net

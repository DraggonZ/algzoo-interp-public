"""Hand-built v24 (17 neurons, a validation build): v23 with the runner-up mechanism ARC's near ties point to.

Research notes: research/20261017_handbuilt/superposition/twin/README.md.

v23 decides near ties exactly, ARC does not. In ARC the number that was runner-up when a contender arrived moves the
second-over-third lead 7-9 times more than other smaller numbers (twin/arc_leak_source.log), as ARC's keep gate (neuron 5)
compares each number with the runner-up so far. v24 adds that, with constants read off ARC:

  runner-up neuron Ru (the 17th): relu(g2 (m2 - x)); it keeps the runner-up so far exactly (m2 = x + Ru/g2 - jump).
  keep gate G: beta relu(x - theta m2), at the number's own step, m2 the runner-up before it; theta 0.3 (ARC's leak on
      that runner-up, 0.073 per unit, over the keep share 0.25); the linear route L carries it one step, so both routes
      reach memory together. A number at the last step is read through the direct route only.
  memory: fades 0.99 per step (ARC's level mode) and votes are read as faded, not compensated (ARC's account: votes fade
      with arrival, later votes are discounted by the runner-up, and the two roughly offset). The shared level and the
      hiding wave are read with the same fading, so they stay exact.
  other constants as the twin: imprint 1 vote, keep share 0.25.

Everything else is v23 (handbuilt_v23.py has the rules). In ARC the runner-up lives in hiding neuron 8 alongside a share
of the hiding wave; fitting it into 16 neurons is the next step.
"""
import numpy as np
from algzoo.handbuilt import Net, FROM_PQ, TO_PQ, _rot
from algzoo.handbuilt_v23 import NM, TURN, wave_coef, wave


def build_v24(theta=0.3, rho=0.99, U_mem=1.0, share=0.25, g2=1e4, w=0.0, A0=1.0, kappa=0.5, shape="forward_safe", beta_f=0.5,
              a=1000.0, s=20.0, g=1e4, lf=None, far=True, sign=-1, level_exact=True, last_gate=False, age_fade=False,
              runner="exact"):
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
    if shape == "no_flip":       # 16 neurons: the wave keeps only the clock's rotating pair, f(k) = cos 45k (peak 1)
        coef = (1.0, 0.0, 0.0)
    elif shape == "no_flip_f1":  # same, with no spill on the next position: f(1) = 0 needs b = -a
        coef = (1.0, -1.0, 0.0)
    else:
        coef = wave_coef(shape, beta_f)
    has_flip = coef[2] != 0
    Ap = A0 * np.hypot(coef[0], coef[1])   # size of the rotating part per unit of K d
    phi = np.arctan2(coef[1], coef[0])     # its phase
    Af = A0 * coef[2]                      # size of the flip part
    if lf is None:                   # L stays positive for x >= -2 m1 with the deepest wave lobe and the looped-back peak
        lf = alpha + 2 * alpha + K * A0 * (1 + w) + K * A0
    U = lf - alpha if U_mem is None else U_mem      # shared memory level per unit of m1
    exact_runner = runner == "exact"
    net = Net(d=(17 if has_flip else 16) - (0 if exact_runner else 1))
    S, D, G, L = (net.add(k) for k in ("store", "detect", "keep", "linear"))
    P = net.add("flip") if has_flip else None
    Ru = net.add("runner") if exact_runner else None
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
    if has_flip:
        # flip: same level and wipe; its own part changes sign each step and gets beta_f K d
        wi[P] = -g
        wh[P, S] = g / s
        wh[P, D] = (g + K * Af / Ap) / a           # the flip neuron's own part is read with weight Ap
        wh[P, P] = -1.0
        add(P, lev)                                     # restores the level the self-weight -1 removed
    flip = {P: 1.0, **{k: -v for k, v in lev.items()}} if has_flip else {}   # flip part = P - clock level

    # runner-up so far: Ru = relu(g2 (m2(t-1) - x_t)); then m2(t) = x_t + Ru / g2 - d_t exactly in every case
    # (below the runner-up; between runner-up and largest: Ru = 0, m2 = x; a new largest: m2 = old largest). From the
    # state of step t-1: m2(t-1) = store / s - clock level / g + Ru / g2.
    if exact_runner:
        m2 = {S: 1 / s, **{k: -v / g for k, v in lev.items()}, Ru: 1 / g2}
        add(Ru, m2, g2)
        wi[Ru] = -g2
    else:
        # runner="store_lag" (16 neurons): the store read without the detector, one step before the number, holds the
        # largest two steps back. Right after a new largest that is the previous largest, the runner-up at that
        # moment; at other steps it is the largest. ARC's hiding neurons 6 and 8 make this comparison (bridge/README.md).
        m2 = {S: 1 / s}
    # keep gate, at the number's own step: beta relu(x_t - theta m2(t-1)), the runner-up before it (ARC's rule)
    wi[G] = beta
    add(G, m2, -beta * theta)
    # linear route: alpha (x - m1) + lf m1 = -alpha level / g + lf m1; minus the looped-back peak w K A0 d;
    # plus (1 - w) x the forward copy of the wave at its own position (offset t - r: clock angle + 90 degrees, flip x -1)
    add(L, {k: -alpha / g * v for k, v in lev.items()})
    wh[L, G] += 1.0                                 # carries the gated route one step, so both reach memory together
    add(L, m1, lf)
    wh[L, D] += -w * K * A0 / a
    wh[L, C] += (1 - w) * Ap * np.array([np.cos(phi - 2 * TURN), np.sin(phi - 2 * TURN)]) @ TO_PQ
    add(L, flip, -(1 - w) * Ap)

    # memory: shift, loop back, writes
    for j in range(1, NM):
        wh[Y[j], Y[j - 1]] = rho
    wh[Y[0], Y[NM - 1]] = w
    wh[Y[0], L] += sign
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
        f2 = rho ** 2 if age_fade else 1.0           # age_fade: every vote fades rho per step since it arrived (two steps
        wo[p, Y[7 - p]] += sign * f2                # pass before a number reaches memory), position 8 by one step
        add(p, {S: 1 / s}, -sign * U * rho ** (7 - p) * f2, wo)   # the shared level, faded like the slot
        hide(p, rho ** (7 - p) * f2)                # the wave copy in memory has faded as much
    f1 = rho if age_fade else 1.0
    wo[8, L] += f1
    add(8, {S: 1 / s}, -(lf - alpha) * f1, wo)       # L holds alpha x + (lf - alpha) m1(8)
    hide(8, (1 - w) * f1)                           # its copy carries 1 - w of the wave; no looped-back share
    # position 9: A0 x9 - A0 K d9 (a record at the last step is hidden like any other)
    if last_gate:     # the last number gets the runner-up discount too: alpha x9 + G(9), so it is judged like the others
        add(9, xprev, alpha, wo)
        wo[9, G] += 1.0
    else:             # direct route only, at full weight (ARC: the last number bypasses the keep gate)
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

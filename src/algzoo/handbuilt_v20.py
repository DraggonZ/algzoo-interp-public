"""Hand-built v20: a robust twin built on the lessons of research/20261017_handbuilt/robust_twin/DESIGN.md.
22 neurons (extra neurons allowed for a robust proof of principle, 2026-10-18). Every weight follows from a
rule written next to it; nothing is fitted.

Why v18 is fragile, in one line: its timer neurons carry a large level (g (m1 - x)) under a small wave, and noise
scales with a neuron's whole value. v20 keeps every timer level about as large as the wave it guards. To do that the
timer may not depend on the current input, so it is wiped one step after a record (through the detector), and a small
reveal group, silent except at records, undoes the hiding when the largest arrives at the last step.

Notation: m1(t) is the largest so far after input t (0 before any positive number); a record at step r has jump
d = x_r - m1(r - 1); A0 is the vote per unit of value; K = 1 + kappa.

Neurons
  S (store)      S_t = s m1(t-1).
  D (detector)   D_t = a d_t, nonzero only at a record.
  Zc (carrier)   Zc_t = d_(t-1): the previous step's jump, unconditionally.
  Z (flag)       Z_t = relu(d_(t-1) + w (x_t - m1(t-1))): equals d_(t-1) only when x_t is a record too (two records in a
                 row), plus w d_t, cancelled where Z is read. Used at the last step only.
  G (keep gate)  G_t = relu(A0 x_t): every positive number's full vote, records included, in the same step.
  C0-C2 (clock)  level LP m1(t-1) plus a 2-D phase that turns 45 degrees per step. Wiped one step after a record
                 (input -W D/a), and two steps after it receives the record's jump from the carrier, K d at angle 0.
                 The level never depends on the input, so it only needs to cover the wave: LP = K (the wave is at most K m1).
  F (flip)       level LP m1(t-1) plus a phase that changes sign each step; same wipe; receives beta K d. Its phase is
                 read as F - mean(C), which stays right after a wipe (both are then 0).
  QC0-QC2, QF    reveal group: the same pre-activations as C and F plus GQ (x_t - m1(t-1)). Silent for a number below
                 the largest (when GQ (m1 - x) exceeds the level and wave); at a record they equal the timer's would-be
                 values plus GQ d, a common shift the phase readout ignores. Read only at the last step: when the
                 largest arrives at step 9 they hand back exactly the hiding the timer still holds.
  Y0-Y8 (memory) a delay line; at step t slot j holds position t - 1 - j, with level lam m1 (the newest slot from the
                 store, older ones by adding each jump). Y0 receives the vote G. Y1 receives the compensation for its
                 position: the timer's wave value there (the timer at that point already holds the right record's
                 phase, or is wiped, which is right for the record's own slot). A record's detector pulse writes its
                 wave's backward half into the slots of earlier positions.

Readout (step 9): positions 0-7 read their slot (Y_(8-p)) minus lam m1 minus the timer's wave plus the reveal group's
wave; minus the backward half of a record at step 8, which the wiped timer no longer hides (through the carrier).
Position 8 reads Y0 minus K A0 d_8 if a record was at 8 (carrier), plus it back if a record at 9 overtook it (flag).
Position 9 reads G minus a large multiple of D (a record at step 9 is never named).

What follows: a hidden largest shows A0 (p - kappa d) at its position, where p is the previous largest, so it trails
an overtaken record by A0 kappa d; once overtaken it shows A0 x exactly. Kept numbers show A0 x, plus the spill of
overtaken records' waves.
"""
import numpy as np
from algzoo.handbuilt import FROM_PQ, TO_PQ, _rot

TURN = np.pi / 4
NAMES = ["S", "D", "Zc", "Z", "G", "C0", "C1", "C2", "F", "QC0", "QC1", "QC2", "QF"] + [f"Y{j}" for j in range(9)]


def build_v20(A0=10.0, beta=0.5, kappa=0.5, a=10.0, s=10.0, W=1e4, GQ=None, w=None, lev=1.0, lam=None, big=1e3):
    n = len(NAMES); I = {k: i for i, k in enumerate(NAMES)}
    wi = np.zeros(n); wh = np.zeros((n, n)); wo = np.zeros((10, n))
    S, D, Zc, Z, G, F, QF = (I[k] for k in ("S", "D", "Zc", "Z", "G", "F", "QF"))
    C = [I[f"C{k}"] for k in range(3)]; QC = [I[f"QC{k}"] for k in range(3)]; Y = [I[f"Y{j}"] for j in range(9)]
    K = 1 + kappa; A = A0 / (1 + beta); LP = lev * K
    GQ = 10 * LP if GQ is None else GQ
    lam = 0.81 * K * A0 if lam is None else lam     # covers the deepest negative compensation, 0.805 A0 K m1
    w = GQ if w is None else w                        # the flag only has to tell a record from a number just below m1
    m1 = {S: 1 / s, D: 1 / a}                       # m1(t-1), read from h_(t-1)
    R = _rot(TURN); e1 = np.array([1.0, 0.0])

    # store and detector
    wh[S, S] = 1.0; wh[S, D] = s / a
    wi[D] = a; wh[D, S] = -a / s; wh[D, D] = -1.0
    # carrier and flag
    wh[Zc, D] = 1 / a
    wh[Z, D] = 1 / a; wi[Z] = w
    for j, v in m1.items():
        wh[Z, j] += -w * v
    # keep gate: the full vote of every positive number, in the same step
    wi[G] = A0

    def timer_rows(rows, flip_row, extra_input):
        """Pre-activation of the clock (rows) and flip (flip_row): level, rotation, wipe, injection."""
        for r_ in rows + [flip_row]:
            for j, v in m1.items():
                wh[r_, j] += LP * v                      # level LP m1(t-1)
            wh[r_, D] += -W / a                          # wiped one step after a record
            wi[r_] += extra_input
            for j, v in m1.items():
                wh[r_, j] += -extra_input * v            # extra_input (x_t - m1(t-1)); zero for the timer itself
        wh[np.ix_(rows, C)] += FROM_PQ @ R @ TO_PQ       # the phase turns 45 degrees
        wh[rows, Zc] += K * FROM_PQ @ e1                 # two steps after a record: phase K d at angle 0
        wh[flip_row, F] += -1.0                          # flip phase: -(F - mean C) ...
        wh[flip_row, C] += 1.0 / 3                       # ... (mean C carries the level LP m1(t-2), 0 after a wipe)
        wh[flip_row, Zc] += beta * K                     # two steps after a record: flip phase beta K d

    timer_rows(C, F, 0.0)
    timer_rows(QC, QF, GQ)

    # memory
    for j, v in m1.items():
        wh[Y[0], j] += lam * v                           # level lam m1(t-1)
    wh[Y[0], G] += 1.0                                   # the vote of position t - 1
    for j in range(1, 9):
        wh[Y[j], Y[j - 1]] = 1.0
        wh[Y[j], D] += lam / a                           # level follows each jump
        wh[Y[j], D] += A * K * (np.cos(TURN * j) + beta * (-1.0) ** j) / a   # backward half of a record's wave
    # forward compensation in Y1 (position t - 2), from the timer at t - 1: clock part and flip part
    wh[Y[1], C] += A * (R.T @ e1) @ TO_PQ
    wh[Y[1], F] += -A
    wh[Y[1], C] += A / 3

    # readout
    def wave_read(p, sign, rows, flip_row):
        v = np.linalg.matrix_power(R, 7 - p) @ e1
        wo[p, rows] += sign * A * v @ TO_PQ
        fs = (-1.0) ** (7 - p)
        wo[p, flip_row] += sign * A * fs
        wo[p, rows] += -sign * A * fs / 3
    for p in range(8):
        wo[p, Y[8 - p]] += 1.0
        wo[p, S] += -lam / s                             # minus the slot level lam m1(8)
        wave_read(p, -1.0, C, F)                         # hide the current largest
        wave_read(p, +1.0, QC, QF)                       # unless it was overtaken at the last step
        wo[p, Zc] += -A * K * (np.cos(TURN * (8 - p)) + beta * (-1.0) ** (8 - p))   # a record at 8: its backward half
    wo[8, Y[0]] += 1.0; wo[8, S] += -lam / s
    wo[8, Zc] += -K * A0                                 # a record at 8 is hidden ...
    wo[8, Z] += K * A0; wo[8, D] += -K * A0 * w / a      # ... unless a record at 9 overtook it
    wo[9, G] += 1.0; wo[9, D] += -big
    return wi, wh, wo

"""Chained readout rule: fair-vote push from the row below, and the last row from the readout's zero column sums.

Two mechanisms.

1. Fair votes as a one-step push (as rules/shift_pairs.py and rules/shift_joint.py). A number at position k is seen
   at the end one step younger than a number at position k - 1, so for the two to be read alike, row k must read the
   state as row k - 1 reads it one step later:
       W_oh[k, TGT] = g * W_oh[k - 1, SRC] @ W_hh[SRC, TGT],   k = 1 .. 8,
   SRC = the neurons on in the most common on/off pattern (all but detector 4 and gates 5, 9), TGT = SRC plus the keep
   gate 5. Chained: rows are described in increasing order, so row k - 1 is ARC's own row (already described), not a
   prediction, and errors do not compound. (A pull towards an anchor row above, as in shift_joint.py, only hurts once
   the row below is ARC's: it inverts the push; LEDGER r3a-1.)

2. Zero column sums. Softmax ignores a shift common to all ten readout rows, and the cross-entropy gradient on the
   logits sums to zero over the ten positions, so training barely moves each readout column's sum over positions
   away from its small initial value. ARC's column sums are -1.1 to +0.6 against column sizes 0.9-4.4. So the last
   row described, row 9 (all 16 columns, including the detector's and late gate's, which the push cannot reach), is
       W_oh[9, :] = -(W_oh[0, :] + ... + W_oh[8, :])     (ARC's rows, all described before it).
   Row 9 follows the push badly (error 0.84), but this relation to 0.27.

One constant: the shared gain g.

ORDER: out1, ..., out8, out9. Row 0 and the columns 4 and 9 of rows 1-8 are uncovered (ARC's, stated anyway).

STRUCTURE_BITS (31):
  push form (row pushed one step forward): 2; its on-pattern (4, 5, 9 off): 2; columns covered (six units in/out): 6
  which rows the push covers (1-8 of 1-9): log2 C(9, 8) = 3, plus their number: 3
  chain order: 1 bit per covered row (9)
  which row is the zero-sum row (1 of 10): 4; its form (minus the sum of all other rows): 1
  one shared gain: 1
Selection note: the push pattern (rows 1-8) was picked from 511 chained patterns on a quick evaluator (r3a-1), about
9 bits of search against 6 charged.
"""
import numpy as np
import torch as th
from algzoo.core import example

WI, WH, WO = example()
SRC = [i for i in range(16) if i not in (4, 5, 9)]
TGT = [0, 3, 10, 11, 12, 13, 14, 15, 1, 6, 7, 8, 2, 5]
PUSH_ROWS = list(range(1, 9))
SUM_ROW = 9
STRUCTURE_BITS = 31
KINDS = np.array(["gain"])
COVERED = [f"{i}->out{k}" for k in PUSH_ROWS for i in TGT] + [f"{i}->out{SUM_ROW}" for i in range(16)]
ORDER = [f"out{k}" for k in PUSH_ROWS] + [f"out{SUM_ROW}"]

_n = len(TGT)
_B = np.zeros((_n, _n)); _B[[TGT.index(i) for i in SRC], :] = WH[np.ix_(SRC, TGT)]
_PUSHED = th.tensor(np.stack([WO[k - 1, TGT] @ _B for k in PUSH_ROWS]).ravel())
_SUM = th.tensor(-WO[[k for k in range(10) if k != SUM_ROW]].sum(0))


def init_constants():
    return np.array([0.92])


def weights(C):
    return th.cat([C[0] * _PUSHED, _SUM])


def fit(C0, target, F):
    """Read g off ARC's covered push weights by plain least squares (unweighted; ARC's weights only, no answers).
    The scorer's default curvature-weighted fit gives g = 1.03 and net +82.5 (LEDGER r3a-4); this one about 0.92."""
    m = len(PUSH_ROWS) * _n
    p, t = _PUSHED.numpy(), np.asarray(target)[:m]
    return np.array([p @ t / (p @ p)])


def counterparts(C):
    return [("g", float(C[0]), None)]

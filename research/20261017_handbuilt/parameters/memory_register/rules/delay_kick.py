"""Round 5 (r5a): rules/delay_line.py plus the record's kick into the delay line.

0b. Shared kick (new in round 5). Neuron 10 is neuron 11 one step later (0. below), and it receives the record's write
   at the same moment as neuron 11: the detector (neuron 4, on only in the state that reads a new record) feeds both
   with the same weight. So one step after a record both start from the same kick, and from then on neuron 10 carries
   neuron 11's trace one step later (ARC's activity, by age of the largest: h10(age a) is about 1.1 h11(age a-1)
   for a >= 2, and both get about 11 from the detector at age 1). Rule: W[10, 4] = W[11, 4], ARC's weight into
   neuron 11 (row 11 is uncovered, so it is stated anyway). ARC: 1.76 against 1.71 (3%). No constant.
   KICK env var (default "4") lists the outside columns of row 10 copied from row 11; only 4 is the rule (looks).
   Its structure: 4 bits (which outside source of 8: 3; form "same as row 11": 1).
   Judgement call: the 1.76 / 1.71 match was seen by eye in the weights before the story was written; the pair was
   not searched for (it is the delay pair), but the column was.

Round 4 (r4b): rules/chain_push_sum.py (readout, unchanged) plus one memory-block row from a delay line.

0. Delay line in the memory (new). Within the memory block, neuron 10 reads only neuron 11, with weight 1: it is
   neuron 11 one step later. That is why it switches off one age of the largest later (6 against 5). Row 10's eight
   block weights (from 0, 3, 10-15) = (0, 0, 0, 1, 0, 0, 0, 0); ARC's are 1.10 from 11 and rms 0.07 for the rest.
   No constant. Described first in ORDER (in10), so the readout push, which reads every memory row, can use ARC's
   row 10. Row 10's other incoming weights (detector 1.76, hiding, input) stay uncovered.
   Its own structure: 7 bits (which row of 8: 3; which source of 8: 3; form "single source, weight 1": 1).
   DELAY env var (default "10:11") lists dst:src links for looks; only 10:11 is the rule (11:12 failed, LEDGER r4b-3).
   Note: the default fit's gain moves from 1.03 to 0.92 when row 10 is covered (curvature cross-terms between row 10
   and the readout rows). At equal gain the delay row is worth about +6 bits; see LEDGER r4b-2.

Readout part, as rules/chain_push_sum.py. Two mechanisms.

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
import os
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

MEM = [0, 3, 10, 11, 12, 13, 14, 15]
DELAY = [tuple(int(a) for a in x.split(":")) for x in os.environ.get("DELAY", "10:11").split(",") if x]
_DL = th.tensor([1.0 if i == src else 0.0 for dst, src in DELAY for i in MEM], dtype=th.float64)
COVERED = [f"{i}->{dst}" for dst, src in DELAY for i in MEM] + COVERED
ORDER = [f"in{dst}" for dst, src in DELAY] + ORDER
STRUCTURE_BITS = STRUCTURE_BITS + 7 * len(DELAY)
KICK = [int(c) for c in os.environ.get("KICK", "4").split(",") if c]
_KICK = th.tensor([WH[11, c] for c in KICK], dtype=th.float64)
_DL = th.cat([_DL, _KICK])
COVERED = [f"{i}->{dst}" for dst, src in DELAY for i in MEM] + [f"{c}->10" for c in KICK] + COVERED[8 * len(DELAY):]
STRUCTURE_BITS = STRUCTURE_BITS + 4 * len(KICK)
# looks only (LEDGER r5a-3): outside columns of row 10 stated as zero ("10 reads nothing else"); "in" = input weight
ZERO = [c for c in os.environ.get("ZERO", "").split(",") if c]
_DL = th.cat([_DL, th.zeros(len(ZERO), dtype=th.float64)])
COVERED = COVERED[:8 * len(DELAY) + len(KICK)] + [("input" if c == "in" else c) + "->10" for c in ZERO] + \
    COVERED[8 * len(DELAY) + len(KICK):]
STRUCTURE_BITS = STRUCTURE_BITS + len(ZERO)
_n = len(TGT)
_B = np.zeros((_n, _n)); _B[[TGT.index(i) for i in SRC], :] = WH[np.ix_(SRC, TGT)]
_PUSHED = th.tensor(np.stack([WO[k - 1, TGT] @ _B for k in PUSH_ROWS]).ravel())
_SUM = th.tensor(-WO[[k for k in range(10) if k != SUM_ROW]].sum(0))


def init_constants():
    return np.array([0.92])


def weights(C):
    return th.cat([_DL, C[0] * _PUSHED, _SUM])


def counterparts(C):
    return [("g", float(C[0]), None)]


if os.environ.get("FIXG"):   # analysis only (r4b quick looks): hold the gain fixed to compare at equal g
    def fit(C0, target, F):
        return np.array([float(os.environ["FIXG"])])

"""r7a: the hiding neurons as comparators on the detector's scale. Unchained; one constant.

Every comparing neuron's pre-activation is w_in,j (x - reference). The detector (4) builds its reference from the
store and itself: W[4,2]/w_in,4 = -0.0485, W[4,4]/w_in,4 = -0.097. The hiding rows reuse that reference.

1. Neuron 7 is the detector's mirror. Row 7 (all 16 recurrent weights) = (w_in,7 / w_in,4) x row 4, a factor -1.30.
   So neuron 7 = ReLU(-1.30 x the detector's pre-activation): the detector carries "arriving number above the
   largest so far", neuron 7 the same comparison below zero. No constant.
2. Neurons 1, 6, 8 compare on the detector's scale: W[j,2] = w_in,j x W[4,2]/w_in,4 (the store part of the shared
   reference). No constant.
3. Blindness inside the hiding group: each of rows 1, 6, 8 reads the hiding group at right angles to the hiding input
   weights, so the previous number cancels; the weight from neuron 7 (largest input weight) is fixed by the row's
   other three hiding weights (uncovered). No constant.
4. Which largest: 6 and 8 compare with the previous largest, so they read no detector (W[6,4] = W[8,4] = 0); neuron 1
   reads kappa times neuron 7's fresh-largest weight, W[1,4] = kappa w_in,1 W[4,4]/w_in,4 (kappa the one constant).
5. Gates: the late gate's push reaches 1 and 6 only, the keep gate's write 1 only: 7 and 8 read neither gate, 6 not
   the keep gate (5 weights = 0; for row 7 these replace the mirror's values).
6. Neurons 1 and 6 take the same memory correction to their reference: W[1,m] = (w_in,1/w_in,6) W[6,m] for the
   8 memory neurons (row 6's memory weights are not covered).
Uncovered: input weights, rows 1, 6, 8's other hiding weights and late-gate weights into 1 and 6, row 6 and 8's memory
weights, neuron 1's keep-gate weight. All weights derived from ARC come from example() at load; no states used.

STRUCTURE_BITS = 24: mirror (which row among the 12 non-hiding neurons: 3.6, form: whole row scaled by input ratio: 1)
5; store ratio on rows 1, 6, 8: 3; blindness (which column: 2, applied per row: 3) 5; detector per row 1, 6, 8
(none / kappa): 3; gate zeros (5 of the 8 gate weights into the four: log2 56 = 5.8) 6; memory shared 1 <- 6 (which
partner among 6, 8: 1, form: 1) 2. Judgement call: the choice among about 140 option combinations (explorer,
`../combos.py`) was made on the explorer's pricing; log2 140 = 7 bits is about what the per-row counts above add up
to, so no extra charge.
"""
import numpy as np
import torch as th
from algzoo.core import example

WI, WH, WO = example()
M = [0, 3, 10, 11, 12, 13, 14, 15]
H = [1, 6, 7, 8]
RS, RD = WH[4, 2] / WI[4], WH[4, 4] / WI[4]


def _blind(j, i=7):
    return -sum(WH[j, k] * WI[k] for k in H if k != i) / WI[i]


FIXED = {}
FIXED.update({f"{i}->7": WI[7] / WI[4] * WH[4, i] for i in range(16)})
for j in (1, 6, 8):
    FIXED[f"2->{j}"] = RS * WI[j]
    FIXED[f"7->{j}"] = _blind(j)
FIXED.update({"4->6": 0.0, "4->8": 0.0, "9->8": 0.0, "5->8": 0.0, "5->6": 0.0, "9->7": 0.0, "5->7": 0.0})
FIXED.update({f"{m}->1": WI[1] / WI[6] * WH[6, m] for m in M})
COVERED = list(FIXED) + ["4->1"]
_fixed = th.tensor([FIXED[n] for n in FIXED], dtype=th.float64)
KINDS = np.array(["mix"])
STRUCTURE_BITS = 24


def init_constants():
    return np.array([1.0])   # start at kappa = 1 (neuron 1 reads the fresh largest); the default fit sets it


def weights(C):
    C = th.as_tensor(C, dtype=th.float64)
    return th.cat([_fixed, (C[0] * WI[1] * RD).reshape(1)])


def counterparts(C):
    return [("kappa", float(C[0]), None)]

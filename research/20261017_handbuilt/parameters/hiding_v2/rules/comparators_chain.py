"""r7a: ../../storegate_rules/rules/overwrite_store.py (+117.9 under v2, chained) plus the hiding neurons' rows as
comparators on the detector's scale. Chained; no new constant.

Every comparing neuron's pre-activation is w_in,j (x - reference). The detector (4) builds its reference from the
store and itself (W[4,2]/w_in,4 = -0.0485, W[4,4]/w_in,4 = -0.097). The hiding rows reuse that reference.

1. Neuron 7 is the detector's mirror: row 7 = (w_in,7 / w_in,4) x ARC's row 4, a factor -1.30, so neuron 7 =
   ReLU(-1.30 x the detector's pre-activation). The detector carries "arriving number above the largest so far",
   neuron 7 the same comparison below zero.
2. Neurons 6 and 8 compare on the detector's scale with the previous largest: store weight w_in,j W[4,2]/w_in,4,
   detector weight 0, memory weights w_in,j / w_in,4 x the detector's memory weights (the same memory correction to
   the reference as the detector).
3. Neuron 1 compares on the same scale (store weight as in 2) and takes neuron 6's memory correction:
   W[1,m] = (w_in,1 / w_in,6) x ARC's W[6,m] (row 6 is described before row 1). Its detector weight stays uncovered.
4. Blindness inside the hiding group: rows 1, 6, 8 read the hiding group at right angles to the hiding input weights,
   so the previous number cancels; the weight from neuron 7 is fixed by the row's other three (uncovered).
5. Gates: the late gate's push reaches 1 and 6 only, the keep gate's write 1 only: 7 and 8 read neither gate, 6 not
   the keep gate (5 weights = 0; for row 7 these replace the mirror's values).
Order: in4 (the base rule's detector row), in7, in6, in8, in1, then the base rule's in10 and readout rows (its readout
push reads ARC's hiding rows, which are then already described). No states are used.

STRUCTURE_BITS = base 48 + 26: mirror (which row among the 12 non-hiding neurons 3.6, form 1) 5; store ratio on rows
1, 6, 8: 3; blindness (which column 2, per row 3) 5; detector weight 0 for 6, 8 and uncovered for 1: 3; gate zeros
(5 of 8 gate weights: log2 56 = 5.8) 6; memory source per row 1, 6, 8 (detector / neuron 6): 3; chain links in7,
in6, in8, in1: 4 -> 29. Judgement call: the options were compared on the explorer's pricing (`../combos.py`,
`../explore.py`); their count is about what the per-row bits above charge.
"""
import importlib.util
from pathlib import Path
import numpy as np
import torch as th
from algzoo.core import example

_p = Path(__file__).resolve().parents[2] / "storegate_rules" / "rules" / "overwrite_store.py"
_spec = importlib.util.spec_from_file_location("overwrite_store_base", _p)
B = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(B)

WI, WH, WO = example()
M = [0, 3, 10, 11, 12, 13, 14, 15]
H = [1, 6, 7, 8]
RS = WH[4, 2] / WI[4]


def _blind(j, i=7):
    return -sum(WH[j, k] * WI[k] for k in H if k != i) / WI[i]


FIXED = {}
FIXED.update({f"{i}->7": WI[7] / WI[4] * WH[4, i] for i in range(16)})
for j in (6, 8):
    FIXED.update({f"{m}->{j}": WI[j] / WI[4] * WH[4, m] for m in M})
FIXED.update({f"{m}->1": WI[1] / WI[6] * WH[6, m] for m in M})
for j in (1, 6, 8):
    FIXED[f"2->{j}"] = RS * WI[j]
    FIXED[f"7->{j}"] = _blind(j)
FIXED.update({"4->6": 0.0, "4->8": 0.0, "9->8": 0.0, "5->8": 0.0, "5->6": 0.0, "9->7": 0.0, "5->7": 0.0})
NEW = list(FIXED)
_new = th.tensor([FIXED[n] for n in NEW], dtype=th.float64)

COVERED = NEW + list(B.COVERED)
ORDER = ["in4", "in7", "in6", "in8", "in1"] + [r for r in B.ORDER if r != "in4"]
STRUCTURE_BITS = B.STRUCTURE_BITS + 29
KINDS = B.KINDS
init_constants = B.init_constants
if hasattr(B, "fit"):
    fit = B.fit


def weights(C):
    w = B.weights(C)
    return th.cat([_new.to(w.dtype), w])


def counterparts(C):
    return B.counterparts(C)

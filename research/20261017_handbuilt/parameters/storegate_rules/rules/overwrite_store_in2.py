"""r6b-3: rules/overwrite_store.py plus two zero-constant statements on the store's row (row 2), described first.

(a) The store hears numbers only through the detector: input->2 = 0 (ARC 0.015). This is why the store is one step
    behind the detector (the store's gain 17.6 = W[2,4] x input->4 comes entirely through neuron 4).
(b) Blindness: the store reads the hiding neurons at right angles to their input weights, so the arriving number
    cancels: sum_h W[2,h] WI[h] = 0 over h in 1, 6, 7, 8. Fixes 8->2 from ARC's uncovered 1->2, 6->2, 7->2 and the
    hiding input weights (0.1548 against ARC 0.1545).
Chain: in2 first (depends only on uncovered weights), then in4 (reads ARC's row 2, described before it), then
delay_kick's rows.

STRUCTURE_BITS: overwrite_store's, plus 5: (a) which weight is zero (the input weight of row 2): 1; (b) blindness
form: 1, which hiding weight is solved (1 of 4): 2; chain link: 1.
"""
import importlib.util
from pathlib import Path
import torch as th
from algzoo.core import example

_spec = importlib.util.spec_from_file_location("overwrite_store_base", Path(__file__).resolve().parent / "overwrite_store.py")
B = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(B)

WI, WH, WO = example()
_B8 = -(WH[2, 1] * WI[1] + WH[2, 6] * WI[6] + WH[2, 7] * WI[7]) / WI[8]
_IN2 = th.tensor([0.0, _B8], dtype=th.float64)

COVERED = ["input->2", "8->2"] + list(B.COVERED)
ORDER = ["in2"] + list(B.ORDER)
STRUCTURE_BITS = B.STRUCTURE_BITS + 5
KINDS = B.KINDS
init_constants = B.init_constants
counterparts = B.counterparts


def weights(C):
    return th.cat([_IN2, B.weights(C)])

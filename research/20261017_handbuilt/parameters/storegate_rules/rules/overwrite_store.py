"""r6b: ../memory_register/rules/delay_kick.py (+109.9 under v2; readout push, zero column sums, delay line, kick),
plus the detector's row from two store/detector mechanisms. Chained.

New part, row 4 (the detector's 16 recurrent weights), no constant:

1. The detector reads the fresh store. Store (2) and detector (4) read the same state h_t; the detector's
   recurrent drive is -c times the store's own recurrent pre-activation, W[4, i] = -c W[2, i] for all 16 i, so
   d_{t+1} = ReLU(k x_t - c s_{t+1}): the arriving number is compared with what the store is about to hold.
2. A record overwrites the store. When the detector fires, the store's next value is
       s_{t+2} = lam s_{t+1} + a d_{t+1} = (lam - a c) s_{t+1} + a k x_t,     lam = W[2,2], a = W[2,4],
   so the old content is wiped exactly when c = lam / a. With 1, the store/detector block is lam times a nilpotent
   2 x 2 block (4->4 = -2->2). This fixes c from ARC's store row: c = W[2,2] / W[2,4] = 0.969 / 1.731 = 0.560
   (least squares on the row: 0.552).
Row 2 (the store) and input->4 stay uncovered (ARC's, stated anyway); row 4 depends only on them. Described first.

STRUCTURE_BITS: delay_kick's 108, plus 6 for the new part: which row the detector reads (the store, 1 of 3 other
neurons of the store/detector/gates set): 2; form (minus a multiple of that row, all 16 columns, input left out): 1;
the multiple from the overwrite condition (ratio of the store's self and detector weights): 2; chain link: 1.
Judgement call: the relation "row 4 = -0.55 row 2" was known before (parameters README experiment 9); the
overwrite condition was derived before checking its value (0.560).
"""
import importlib.util
import os
from pathlib import Path
import numpy as np
import torch as th
from algzoo.core import example

_p = Path(__file__).resolve().parent.parent.parent / "memory_register" / "rules" / "delay_kick.py"
_spec = importlib.util.spec_from_file_location("delay_kick_base", _p)
B = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(B)

WI, WH, WO = example()
_C = WH[2, 2] / WH[2, 4]
_ROW4 = th.tensor(-_C * WH[2], dtype=th.float64)
NEW = [f"{i}->4" for i in range(16)]

COVERED = NEW + list(B.COVERED)
ORDER = ["in4"] + list(B.ORDER)
STRUCTURE_BITS = B.STRUCTURE_BITS + 6
KINDS = B.KINDS
init_constants = B.init_constants


def weights(C):
    return th.cat([_ROW4, B.weights(C)])


def counterparts(C):
    return B.counterparts(C)

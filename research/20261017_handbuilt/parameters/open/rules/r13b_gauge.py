"""r13b: r12a_regpush.py (+172.5 under v2, chained) with the push's common offset fixed by the readout's zero column sums.

Readout rows 1-8 (all 16 columns):  row_k = g Push(row_{k-1}) - row_0 / 9.

Mechanism. The readout feeds a softmax over the ten positions, so adding the same score to every position changes
nothing. Fair votes (row k reads the final state as row k - 1 reads it one step later) therefore fix each row only up
to an offset d common to all pushed rows: row_k = g Push(row_{k-1}) + d, k = 1..9. ARC's readout has zero column sums
(the rule already states that for row 9), and summing the nine relations gives the offset:
    sum_{k=1..9} row_k = -row_0 = g Push(sum_{k=0..8} row_k) + 9 d = -g Push(row_9) + 9 d
    =>  d = (g Push(row_9) - row_0) / 9.
Row 9 is described after rows 1-8 (from the zero sums of rows 0-8), so rows 1-8 cannot read it; its term is dropped,
leaving d = -row_0 / 9. Readout row 0 is uncovered, so the reader holds it. Measured (`../r13b_look4.py`): along row 0
the identity gives -0.097 x row 0, and the mean deficit of the pushed rows is -0.098 x row 0; the dropped term's other
directions are not borne out (with it the rule loses 16 bits, `../r13b_look3.py`). The offset is the zero-sum share
of the oldest position, which has no predecessor to be pushed from, spread evenly over the nine positions that do.

Scale freedom: d is a readout row; each column rescales by 1/s like every other readout weight. 1/9 is a count.

STRUCTURE_BITS = r12a's 83 + 3: a common offset shared equally by the pushed rows (form) 1; its value from the zero
sums over the nine pushed rows 1; the newest row's push term dropped 1. Judgement call: the row-0 regularity was seen
first by round 12's r12b (hand-set c = 0.1); here its value is derived, but the derivation was sought after seeing it.
"""
import importlib.util
from pathlib import Path
import torch as th
from algzoo.core import example

_p = Path(__file__).resolve().parent / "r12a_regpush.py"
_spec = importlib.util.spec_from_file_location("r12a_regpush_base", _p)
B = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(B)

WI, WH, WO = example()
COVERED = list(B.COVERED)
ORDER = list(B.ORDER)
STRUCTURE_BITS = B.STRUCTURE_BITS + 3
KINDS = B.KINDS
init_constants = B.init_constants

_off = th.zeros(len(COVERED), dtype=th.float64)
_row0 = th.tensor(WO[0], dtype=th.float64)
for _j, _n in enumerate(COVERED):
    _src, _dst = _n.split("->")
    if _dst in {f"out{k}" for k in range(1, 9)}:
        _off[_j] = -_row0[int(_src)] / 9


def weights(C):
    return B.weights(C) + _off


def counterparts(C):
    return B.counterparts(C)

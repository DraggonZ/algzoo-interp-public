"""r14a: r13b_gauge.py (+181.3 under v2, chained) with the push's extra step fed the list's smallest number.

Readout rows 1-8 (all 16 columns):  row_k = g Push_min(row_{k-1}) - row_0 / 9, with
    h_11 = ReLU(W_hh h_10 + min(x_0..x_9) w_in),     Push_min(r) = argmin_q E[(q . h_10 - r . h_11)^2].

Mechanism (partly found by search; see `../LEDGER.md` r14a-6). Fair votes: row k reads the final state as row k - 1
reads it one step later, and "one step later" needs a number to arrive. r12a used x_new = 0. Here the newcomer is
the list's own smallest number, which can never be first or second. That reading is only part of the story: the
on/off pattern at the extra step is the same as with 0 (hiding neurons on, detector off), and the change to rows 1-8
is almost all one offset common to the eight rows, mostly in the detector's column. Shifting every position's score
by the same amount is invisible to the softmax, so the push cannot pin that offset; r13b pinned its row-0 part from
the zero sums, and the small arrival supplies another part. Arrivals of about -1 to -1.5 work; drawn below the minimum
(about -2.3) or a fixed -5 is worse. No constant: the arrival is read off the list itself.
States as r12a: FIT_SEED (default 101), 400,000 standard-normal lists of length 10 run through ARC's input and
recurrent weights (all described earlier in the chain or uncovered); no answers, no readout row k.

Measured (explorer, gain refitted; `../r14a_look2.py`..`../r14a_look5.py`): zero +181.4, fresh +179.0, the list's
smallest +197.9 (seed 101; +199.1 seed 102). Refitted gain 0.92 (r13b: 1.03).

STRUCTURE_BITS = r13b's 86 + 5: the arrival chosen among about 18 forms looked at (zero, fresh, scaled fresh x3,
mixtures x3, below the second-largest, below the largest, min(x,0), max(x,0), -|x|, below the minimum, the minimum,
fixed -1, -2, -5): log2 18 = 4.2 -> 5 bits with the form. The base's 1 bit for "0, 1 of 2" is not credited back.
Scale freedom: the arrival is an input value (input-weight units); h_11 rescales with each neuron, so the push is
scale-free as r12a's is.
"""
import importlib.util
import os
from pathlib import Path
import numpy as np
import torch as th
from algzoo.core import example

_p = Path(__file__).resolve().parent / "r13b_gauge.py"
_spec = importlib.util.spec_from_file_location("r13b_gauge_base", _p)
B = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(B)
R12 = B.B   # r12a module, holds the zero-arrival push _REG

WI, WH, WO = example()
N = 400_000
_g = np.random.default_rng(int(os.environ.get("FIT_SEED", "101")))
_x = _g.standard_normal((N, 10))
_h = np.zeros((N, 16))
for _t in range(10):
    _h = np.maximum(_h @ WH.T + _x[:, _t, None] * WI[None, :], 0)
_h11 = np.maximum(_h @ WH.T + _x.min(1)[:, None] * WI[None, :], 0)
_G = _h.T @ _h / N
PUSH_ROWS = list(range(1, 9))
_REG = np.stack([np.linalg.solve(_G, _h.T @ (_h11 @ WO[k - 1]) / N) for k in PUSH_ROWS])   # [8, 16]
del _h, _h11, _x

COVERED = list(B.COVERED)
ORDER = list(B.ORDER)
STRUCTURE_BITS = B.STRUCTURE_BITS + 5
KINDS = B.KINDS
init_constants = B.init_constants

_delta = th.zeros(len(COVERED), dtype=th.float64)
for _j, _n in enumerate(COVERED):
    _src, _dst = _n.split("->")
    if _dst in {f"out{k}" for k in PUSH_ROWS}:
        _k = int(_dst[3:])
        _delta[_j] = _REG[_k - 1, int(_src)] - R12._REG[_k - 1, int(_src)]


def weights(C):
    return B.weights(C) + C[0] * _delta


def counterparts(C):
    return B.counterparts(C)

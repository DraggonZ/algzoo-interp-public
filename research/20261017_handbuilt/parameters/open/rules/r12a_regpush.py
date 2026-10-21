"""r12a: ../../hiding_v2/rules/r9b_chain.py (+161.9 under v2, chained) with the readout push taken through the ReLU.

Readout rows 1-8 (all 16 columns), replacing the base's linearised push on 14 columns:

Fair votes, stated on the network's own states. A number at position k is seen at the end one step younger than a
number at position k - 1, so row k must read the final state h_10 as row k - 1 reads it one step later. The base
wrote "one step later" as a fixed linear block (every neuron but 4, 5, 9 on) times a fitted gain. Here it is the
actual next step through the ReLU, with no fresh number arriving:
    h_11 = ReLU(W_hh h_10),       row_k = argmin_r E[(r . h_10 - row_{k-1} . h_11)^2]      (k = 1 .. 8)
i.e. row k is the least-squares linear readout of h_10 that reproduces what ARC's row k - 1 (already described,
chain order out1..out8) reads one step later; then scaled by the base's one constant g (refitted; it lands near 1).
Each neuron enters the push only when it is on, in the state where it is on, so no on-pattern or column set is
chosen, and the detector's and late gate's columns are covered too. States: FIT_SEED (default 101), 400,000
standard-normal lists of length 10, run through ARC's input and recurrent weights (every recurrent row is either
uncovered or described before the readout rows, so the reader holds them); no answers, no readout row k.
Row 9 stays the zero column sum of ARC's rows 0-8; row 0 stays uncovered.

STRUCTURE_BITS = r9b_chain's 86 - 3 = 83: the base's push charged form 2, on-pattern 2, columns 6 (10 bits). The
regression push: form (least squares through the actual next step) 2, the fresh number at the extra step set to 0
(1 of 2) 1, all 16 columns 1, looked-at variants (fresh number or 0, 14 or 16 columns: 2 bits of search) 2,
choice of state distribution (the task's own lists) 1 -> 7 bits.
"""
import importlib.util
import os
from pathlib import Path
import numpy as np
import torch as th
from algzoo.core import example

_p = Path(__file__).resolve().parents[2] / "hiding_v2" / "rules" / "r9b_chain.py"
_spec = importlib.util.spec_from_file_location("r9b_chain_base", _p)
B = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(B)

WI, WH, WO = example()
N = 400_000
_g = np.random.default_rng(int(os.environ.get("FIT_SEED", "101")))
_x = _g.standard_normal((N, 10))
_h = np.zeros((N, 16))
for _t in range(10):
    _h = np.maximum(_h @ WH.T + _x[:, _t, None] * WI[None, :], 0)
_h11 = np.maximum(_h @ WH.T, 0)
_G = _h.T @ _h / N
PUSH_ROWS = list(range(1, 9))
_REG = np.stack([np.linalg.solve(_G, _h.T @ (_h11 @ WO[k - 1]) / N) for k in PUSH_ROWS])   # [8, 16]
del _h, _h11, _x

_basecov = list(B.COVERED)
_keep = [i for i, n in enumerate(_basecov) if not any(n.endswith(f"->out{k}") for k in PUSH_ROWS)]
_bpush = [i for i, n in enumerate(_basecov) if any(n.endswith(f"->out{k}") for k in PUSH_ROWS)]
NEW = [f"{i}->out{k}" for k in PUSH_ROWS for i in range(16)]
COVERED = [_basecov[i] for i in _keep] + NEW
ORDER = list(B.ORDER)
STRUCTURE_BITS = B.STRUCTURE_BITS - 3
KINDS = B.KINDS
init_constants = B.init_constants
_regt = th.tensor(_REG.ravel(), dtype=th.float64)
_keept = th.tensor(_keep)


def weights(C):
    w = B.weights(C)
    return th.cat([w[_keept], C[0] * _regt])


def counterparts(C):
    return B.counterparts(C)

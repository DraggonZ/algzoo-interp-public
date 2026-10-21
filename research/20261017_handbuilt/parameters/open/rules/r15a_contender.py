"""r15a: r14a_minarrival.py (+192.8 under v2, +197.9 under v2c, chained) with the fair vote asked only of contenders.

Readout rows 1-8 (all 16 columns):  row_k = g Push_k(row_{k-1}) - row_0 / 9, with
    h_11 = ReLU(W_hh h_10 + min(x_0..x_9) w_in)       (r14a's next step, unchanged),
    Push_k(r) = argmin_q E[(q . h_10 - r . h_11)^2  |  x_k is among the three largest of x_0..x_9].

Mechanism. Fair votes (rounds 1-14): readout row k reads the final state as row k-1 reads it one step later, since
after one more step the number at position k sits at position k-1. Until now the vote was asked to hold on every
list, weighted equally. But training shaped the readout only where its answer was in doubt: the cross-entropy
gradient on row k is E[(p_k - y_k) h_10], which is near zero whenever position k is clearly not the answer
(p_k and y_k both near 0). The answer is the second-largest number; the positions whose scores the training had to
place are the largest (it must lose to the second), the second (it must win) and the third (the closest rival from
below). So the fair vote only needs to hold on lists where position k's number is one of the three largest, and the
push regression is restricted to those lists. On the other lists the readout's scores for position k are free to
disagree with the vote, and ARC's weights do disagree there.

Measured (explorers `../r12a_x.py` and `../r15a_xc.py`, gain refitted; `../r15a_look4.log`..`../r15a_look9.log`),
v2c, minimum arrival, no offset, gross of the 5 bits: top 1 -49.6 (the regression sees too few kinds of state),
top 2 +177.5, top 3 +208.5, top 4 +208.2, top 5 +205.6, top 6 +203.1, top 8 +199.8, all ten +197.6.
This rule (with the offset): v2c +204.6 / +205.5 / +206.9 on list seeds 101-103 (r14a +197.9 / +198.3 / +198.4);
v2 +193.3 / +193.9 / +194.8 (r14a +192.8 / +193.1 / +193.1). The gain is row-specific, not a shared offset
(`../r15a_look10.log`).

States as r12a/r14a: FIT_SEED (default 101), 400,000 standard-normal lists of length 10, run through ARC's input
and recurrent weights (all described earlier in the chain or uncovered). The restriction uses only the list itself
(the task's definition of the contenders), not ARC's readout or answers.

STRUCTURE_BITS = r14a's 91 + 5: restricting the push to contenders (form, 1) and which contenders (top 3), chosen
among about 12 forms looked at (cuts top 1, 2, 3, 4, 5, 6, 8, 10; ranks among x_1..x_9; ranks after the step; ranks at
both times; second or third only): log2 12 = 3.6 -> 4 bits.
Scale freedom: the restriction acts on lists, not weights; the push is scale-free as r12a's is.
"""
import importlib.util
import os
from pathlib import Path
import numpy as np
import torch as th
from algzoo.core import example

_p = Path(__file__).resolve().parent / "r14a_minarrival.py"
_spec = importlib.util.spec_from_file_location("r14a_minarrival_base", _p)
B = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(B)

WI, WH, WO = example()
N = 400_000
TOP = 3
_g = np.random.default_rng(int(os.environ.get("FIT_SEED", "101")))
_x = _g.standard_normal((N, 10))
_h = np.zeros((N, 16))
for _t in range(10):
    _h = np.maximum(_h @ WH.T + _x[:, _t, None] * WI[None, :], 0)
_h11 = np.maximum(_h @ WH.T + _x.min(1)[:, None] * WI[None, :], 0)
_rank = (-_x).argsort(1).argsort(1)          # 0 = largest
PUSH_ROWS = list(range(1, 9))
_REG = []
for _k in PUSH_ROWS:
    _m = _rank[:, _k] < TOP                   # lists where x_k is a contender
    _hk = _h[_m]
    _REG.append(np.linalg.solve(_hk.T @ _hk, _hk.T @ (_h11[_m] @ WO[_k - 1])))
_REG = np.stack(_REG)                         # [8, 16]
del _h, _h11, _x, _rank

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
        _delta[_j] = _REG[_k - 1, int(_src)] - B._REG[_k - 1, int(_src)]


def weights(C):
    return B.weights(C) + C[0] * _delta


def counterparts(C):
    return B.counterparts(C)

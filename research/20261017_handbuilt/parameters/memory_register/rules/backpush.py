"""Round 5 (r5a): the readout push read backwards for a few memory-block rows, on top of rules/delay_line.py.

Mechanism. The push ("fair votes", rules/chain_push_sum.py) says readout row k reads the state as row k-1 does one
step later:  r_k[i] = g * sum_j r_{k-1}[j] W[j, i]  for every column i (j over the usually-on neurons SRC).
Component by component, this one equation can be read for the readout weight r_k[i] or for one recurrent weight
W[j, i], not both. Rows 1..8 of the readout used every equation for the readout. Here the memory components
(i a memory neuron) of the first m equations are read for the block instead: for the block rows S,
    sum_{j in S} r_{k-1}[j] W[j, i] = r_k[i] / g - sum_{j in SRC, j not in S} r_{k-1}[j] W[j, i],   k = 1..m,
solved by least squares over k (m >= |S|), with ARC's readout rows 0..m and ARC's other rows on the right. The
readout rows 1..m keep their push for the non-memory columns (hiding 1, 6, 7, 8, store 2, keep gate 5); their memory
columns are uncovered (ARC's, stated anyway). Rows m+1..8 are pushed in full, row 9 from zero column sums, row 10 is
the delay line (all as rules/delay_line.py).

Why it can work where round 4's full backward solve failed: each block row here is solved only from the equations
where the readout row before weighs that neuron strongly, with every other row exact (ARC's), so the solve is
well conditioned. E.g. readout row 0 weighs neuron 14 at -5.1, its largest entry, so equation 1 pins row 14.

Chain (ORDER): in10; out1..out_m (non-memory columns only: they use ARC's readout row before and the non-memory
columns of the recurrent weights, which no covered row holds); the block rows S (they use ARC's readout rows
0..m, whose covered parts are described before them, and uncovered rows); out_{m+1}..out8 (full, they read every
block row); out9.

Env: BP_ROWS (block rows S, default "14"), BP_M (m, default number of rows).

STRUCTURE_BITS = 38 (delay_line.py) + 3 (m of 1..8) + log2 C(7, |S|) (which rows of the seven left) + 1 (form) +
|S| (chain links) + SEARCH (bits for the allocations tried before choosing; set in the ledger).
"""
import math
import os
import numpy as np
import torch as th
from algzoo.core import example

WI, WH, WO = example()
MEM = [0, 3, 10, 11, 12, 13, 14, 15]
SRC = [i for i in range(16) if i not in (4, 5, 9)]
TGT = [0, 3, 10, 11, 12, 13, 14, 15, 1, 6, 7, 8, 2, 5]
NONMEM = [i for i in TGT if i not in MEM]
S = [int(x) for x in os.environ.get("BP_ROWS", "14").split(",") if x]
m = int(os.environ.get("BP_M", str(len(S))))
SEARCH = int(os.environ.get("BP_SEARCH", "0"))
K = list(range(1, m + 1))
PUSH_FULL = list(range(m + 1, 9))
SUM_ROW = 9

COVERED = [f"{i}->10" for i in MEM]
COVERED += [f"{i}->out{k}" for k in K for i in NONMEM]
COVERED += [f"{i}->{j}" for j in S for i in MEM]
COVERED += [f"{i}->out{k}" for k in PUSH_FULL for i in TGT]
COVERED += [f"{i}->out{SUM_ROW}" for i in range(16)]
ORDER = ["in10"] + [f"out{k}" for k in K] + [f"in{j}" for j in S] + [f"out{k}" for k in PUSH_FULL] + [f"out{SUM_ROW}"]
STRUCTURE_BITS = 38 + 3 + round(math.log2(math.comb(7, len(S)))) + 1 + len(S) + SEARCH
KINDS = np.array(["gain"])

# delay line: row 10 within the memory reads only neuron 11, weight 1
_DL = th.tensor([1.0 if i == 11 else 0.0 for i in MEM], dtype=th.float64)
# push matrix through the usually-on neurons (as chain_push_sum.py)
_n = len(TGT)
_B = np.zeros((_n, _n)); _B[[TGT.index(i) for i in SRC], :] = WH[np.ix_(SRC, TGT)]
_PUSH_NONMEM = th.tensor(np.stack([WO[k - 1, TGT] @ _B[:, [TGT.index(i) for i in NONMEM]] for k in K]).ravel()) \
    if K else th.zeros(0, dtype=th.float64)
_PUSH_FULL = th.tensor(np.stack([WO[k - 1, TGT] @ _B for k in PUSH_FULL]).ravel()) if PUSH_FULL else th.zeros(0, dtype=th.float64)
_SUM = th.tensor(-WO[[k for k in range(10) if k != SUM_ROW]].sum(0))
# backward solve: A x = rhs, A[k, s] = r_{k-1}[S[s]], rhs = r_k[MEM] / g - sum_{j in SRC \ S} r_{k-1}[j] W[j, MEM]
_A = th.tensor(np.array([[WO[k - 1, j] for j in S] for k in K]))
_RK = th.tensor(np.array([WO[k, MEM] for k in K]))
_OTHER = [j for j in SRC if j not in S]
_REST = th.tensor(np.array([WO[k - 1, _OTHER] @ WH[np.ix_(_OTHER, MEM)] for k in K]))
_PINV = th.linalg.pinv(_A) if S else None


def init_constants():
    return np.array([0.92])


def weights(C):
    g = C[0]
    parts = [_DL, g * _PUSH_NONMEM]
    if S:
        parts.append((_PINV @ (_RK / g - _REST)).reshape(-1))
    parts += [g * _PUSH_FULL, _SUM]
    return th.cat(parts)


def counterparts(C):
    return [("g", float(C[0]), None)]


if os.environ.get("FIXG"):   # analysis only: hold the gain fixed to compare mechanisms at equal g
    def fit(C0, target, F):
        return np.array([float(os.environ["FIXG"])])

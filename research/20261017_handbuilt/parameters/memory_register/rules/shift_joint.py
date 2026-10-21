"""Fair votes as a one-step shift of the readout, solved jointly between ARC's uncovered anchor rows.

Mechanism. A number at position t is seen at the end through the memory and hiding neurons after 9 - t steps; a
number at position t + 1 after one step fewer. If both are read alike (fair votes), position t + 1's readout is
position t's readout pushed one step through those neurons' recurrent weights:

    W_oh[t + 1, TGT] = g * W_oh[t, SRC] @ W_hh[SRC, TGT],   t = 0 .. 8,

SRC = the neurons on in the most common on/off pattern (all but detector 4 and gates 5, 9); TGT = SRC plus the keep
gate 5 (memory 0, 3, 10-15, hiding 1, 6, 7, 8, store 2, keep gate 5). Same relation as rules/shift_pairs.py.

What is new. shift_pairs.py predicts each covered row from the uncovered row before it only (one-sided, so a row two
steps from its source carries two steps of error). Here every covered row takes part in both relations it appears
in: it is the push of the row before it, and the row after it is its push. The covered rows 1, 2, 3, 4, 6, 7, 8 are
the least-squares solution of all nine relations, with ARC's uncovered rows 0, 5 and 9 held fixed (stated anyway,
free). Each run of covered rows (1-4 and 6-8) is pinned at both ends, like a boundary-value problem, so errors do
not compound from one side. Inverting the push alone is useless (it contracts some directions); the joint solution
uses the backward relation only where it is well conditioned. Nothing here depends on ARC's covered weights.

One constant: the shared gain g (1 if the votes were exactly fair).

STRUCTURE_BITS (22):
  the form (row pushed one step through the block, forward direction): 2
  the push's on-pattern (detector 4, gates 5, 9 off): 2; which columns are covered (six units each in or out): 6
  which rows are covered: log2 C(9, 7) = 5, plus the number of covered rows (1 of 9): 3
  relations up to and including the one into row 9 (vs stopping at 8): 1
  joint least squares over all relations, equal weights (vs one-sided pushes): 1 + 1
  one shared gain: 1
Selection note: the covered rows were chosen among 494 patterns on a quick evaluator (r2a-2 in LEDGER.md), about 9
bits of search, more than the 8 charged for the row choice; the same pattern was r1a's best on other columns.
"""
import numpy as np
import torch as th
from algzoo.core import example

WI, WH, WO = example()
SRC = [i for i in range(16) if i not in (4, 5, 9)]
TGT = [0, 3, 10, 11, 12, 13, 14, 15, 1, 6, 7, 8, 2, 5]
ROWS = [1, 2, 3, 4, 6, 7, 8]
LAST = 9
STRUCTURE_BITS = 22
KINDS = np.array(["gain"])
COVERED = [f"{i}->out{k}" for k in ROWS for i in TGT]

_n, _R = len(TGT), len(ROWS)
_POS = [TGT.index(i) for i in SRC]
_A = np.zeros((_n, _n)); _A[_POS, :] = WH[np.ix_(SRC, TGT)]      # x_t @ _A = push of x_t (x over TGT columns)
_ANCH = {t: WO[t, TGT] for t in range(10) if t not in ROWS}
_A_t = th.tensor(_A)


def _system(g):
    """Rows of M x = r, x = covered rows flattened; one block of _n equations per relation t -> t+1."""
    idx = {k: j for j, k in enumerate(ROWS)}
    Ms, rs = [], []
    for t in range(LAST):
        if t not in idx and t + 1 not in idx:
            continue
        M = th.zeros(_n, _R * _n, dtype=th.float64); r = th.zeros(_n, dtype=th.float64)
        if t + 1 in idx:
            j = idx[t + 1]; M[:, j * _n:(j + 1) * _n] = M[:, j * _n:(j + 1) * _n] + th.eye(_n, dtype=th.float64)
        else:
            r = r - th.tensor(_ANCH[t + 1])
        if t in idx:
            j = idx[t]; M[:, j * _n:(j + 1) * _n] = M[:, j * _n:(j + 1) * _n] - g * _A_t.T
        else:
            r = r + g * (th.tensor(_ANCH[t]) @ _A_t)
        Ms.append(M); rs.append(r)
    return th.cat(Ms), th.cat(rs)


def init_constants():
    return np.array([0.92])


def weights(C):
    M, r = _system(C[0])
    return th.linalg.solve(M.T @ M, M.T @ r)


def counterparts(C):
    return [("g", float(C[0]), None)]

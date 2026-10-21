"""Exact split of a nudge's effect on the final scores into memory writes by source and step (write-or-carry on the memory).

Nudge some inputs by eps (dX). Within the real run's on/off pattern the network is linear, so (for lists whose on/off
pattern the nudge does not change) the change of the memory obeys
    dm_t = D_t (A dm_{t-1} + sum_s B[:, s] do_{s,t-1} + w dx_t),
with D_t the memory's on/off at step t. The final score change is
    dS = sum_t Q_t (writes at t) + W_oh[:, others] do_9,   Q_t = R D_9 A D_8 ... A D_t  (R = memory readout).
So every write (source s at step t) gets its exact share of the final scores. Sources: the eight other neurons
(read at step t-1, written into the memory at step t) and the input itself."""
import numpy as np
from memlib import MEM, OTH, A, B, WIM, RM, RO, run

SOURCES = ["hiding 1", "store 2", "detector 4", "keep gate 5", "hiding 6", "hiding 7", "hiding 8", "late gate 9", "input"]


def decompose(X, dX, eps=1e-4):
    S0, P0, H0 = run(X); S1, P1, H1 = run(X + eps * dX)
    same = ((P0 > 0) == (P1 > 0)).all((1, 2))
    dH = (H1 - H0) / eps; dS = (S1 - S0) / eps
    D = (P0[:, :, MEM] > 0).astype(float)                        # (n, 10, 8)
    n = len(X); Q = np.zeros((n, 10, 10, 8))                      # Q_t per list: (10 positions x 8)
    Q[:, 9] = RM[None] * D[:, 9][:, None, :]
    for t in range(8, -1, -1):
        Q[:, t] = (Q[:, t + 1] @ A) * D[:, t][:, None, :]
    dO = dH[:, :, OTH]                                            # (n, 10, 8) activity change of the other neurons
    C = np.zeros((n, len(SOURCES), 10, 10))                       # source, step of the write, position
    for t in range(10):
        if t > 0:
            u = dO[:, t - 1][:, None, :] * B[None]                # (n, 8 memory, 8 sources)
            C[:, :8, t] = np.einsum("nkm,nms->nsk", Q[:, t], u)
        C[:, 8, t] = np.einsum("nkm,m->nk", Q[:, t], WIM) * dX[:, t][:, None]
    direct = dO[:, 9] @ RO.T                                      # others' own readout at the end
    check = np.abs(C.sum((1, 2)) + direct - dS)[same].max()
    return dict(S0=S0, P0=P0, H0=H0, dH=dH, dS=dS, C=C, direct=direct, same=same, check=check, D=D, Q=Q)

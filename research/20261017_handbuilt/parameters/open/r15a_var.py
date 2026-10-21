"""r15a: variant builder on r12a_regpush. Readout rows 1-8 = g Q_k + off_k (Q, off: [8, 16] arrays); everything
else as r12a (row 9 by zero sums, the recurrent rows of rounds 4-9). States: seed 101, 400,000 lists, ARC's input and
recurrent weights only, as r12a."""
import numpy as np
import torch as th
import r15a_xc as XC
import r12a_x as X2

R12 = X2.load("rules/r12a_regpush.py", "r12a_for_r15a")
WI, WH, WO = X2.WI, X2.WH, X2.WO
PUSH_ROWS = list(range(1, 9))

N = 400_000
import os
_g = np.random.default_rng(int(os.environ.get("FIT_SEED", "101")))
XS = _g.standard_normal((N, 10))
H = np.zeros((N, 16))
for _t in range(10):
    H = np.maximum(H @ WH.T + XS[:, _t, None] * WI[None, :], 0)


def step(h, xnew):
    return np.maximum(h @ WH.T + xnew[:, None] * WI[None, :], 0)


def push(h11, wts=None):
    """Q[k-1] = argmin_q E_w[(q . h_10 - WO[k-1] . h_11)^2]; wts: None or [8, N] list weights per row."""
    Q = []
    for k in PUSH_ROWS:
        w = np.ones(N) if wts is None else wts[k - 1]
        G = (H * w[:, None]).T @ H
        Q.append(np.linalg.solve(G, (H * w[:, None]).T @ (h11 @ WO[k - 1])))
    return np.stack(Q)


class Var:
    def __init__(self, Q, off=None, struct=0):
        self.COVERED = R12.COVERED; self.ORDER = R12.ORDER; self.KINDS = R12.KINDS
        self.STRUCTURE_BITS = R12.STRUCTURE_BITS + struct; self.init_constants = R12.init_constants
        off = np.zeros((8, 16)) if off is None else off
        self.dq = th.zeros(len(self.COVERED), dtype=th.float64); self.do = th.zeros_like(self.dq)
        for j, n in enumerate(self.COVERED):
            s, t = n.split("->")
            if t in {f"out{k}" for k in PUSH_ROWS}:
                k = int(t[3:])
                self.dq[j] = Q[k - 1, int(s)] - R12._REG[k - 1, int(s)]
                self.do[j] = off[k - 1, int(s)]

    def weights(self, C):
        return R12.weights(C) + C[0] * self.dq + self.do


def both(lab, Q, off=None, struct=0):
    v = Var(Q, off, struct)
    a = X2.price(v)[0]; b, d = XC.price(v)
    print(f"{lab:45s} v2 {a:+7.1f}   v2c {b:+7.1f}   gain {d['C'][0]:.3f}", flush=True)
    return a, b

"""Check rule for leak_check.py (not a candidate): the leak of README step 5d, computed live. The memory block is
snap42's (ARC's eigenvectors and shrink factors, angles 1, 2, 3 x 42 deg). The memory's incoming weights from outside
the memory and the readout are set by one least-squares pass on ARC's own states (2^15 lists, seed 1): ARC's
pre-activation of each memory neuron minus the snapped block's part, regressed on ARC's previous states of the other
neurons and the input, on the steps where ARC's neuron is on; the readout regresses ARC's centred final scores on ARC's
final state. These rows depend on ARC's covered weights through ARC's states, so leak_check.py should report a slope
near 1. Same constants and fit as snap42_compensated.py."""
import numpy as np, torch as th
from algzoo.core import example
MEM = [0, 3, 10, 11, 12, 13, 14, 15]
OUT = [i for i in range(16) if i not in MEM]
WI, WH, WO = example()
_ev, _V = np.linalg.eig(WH[np.ix_(MEM, MEM)])
_pairs = sorted([k for k in range(8) if _ev[k].imag > 1e-9], key=lambda k: np.angle(_ev[k]))
_reals = [k for k in range(8) if abs(_ev[k].imag) < 1e-9]
_B = np.concatenate([np.stack([_V[:, k].real, _V[:, k].imag], 1) for k in _pairs] + [_V[:, [k]].real for k in _reals], 1)
ANG = th.tensor(np.radians(42.0) * np.array([1, 2, 3]))
BLOCK = [f"{i}->{j}" for j in MEM for i in MEM]
COMP = [f"input->{j}" for j in MEM] + [f"{i}->{j}" for j in MEM for i in OUT] + [f"{i}->out{k}" for k in range(10) for i in range(16)]
COVERED = BLOCK + COMP
KINDS = np.array(["shrink"] * 3 + ["real"] * 2 + ["basis"] * 64)
STRUCTURE_BITS = 13


def _block(r, re, B):
    blocks = [th.stack([th.stack([r[k] * th.cos(ANG[k]), r[k] * th.sin(ANG[k])]),
                        th.stack([-r[k] * th.sin(ANG[k]), r[k] * th.cos(ANG[k])])]) for k in range(3)]
    return B @ th.block_diag(*blocks, th.diag(re)) @ th.linalg.inv(B)


def init_constants():
    return np.concatenate([[abs(_ev[k]) for k in _pairs], [_ev[k].real for k in _reals], _B.ravel()])


def fit(C0, target, F):
    return C0


def _rows(Wm):
    X = np.random.default_rng(1).standard_normal((2 ** 15, 10))
    h = np.zeros((len(X), 16)); Hp, PRE = [], []
    for t in range(10):
        pre = h @ WH.T + X[:, t, None] * WI; Hp.append(h); PRE.append(pre); h = np.maximum(pre, 0)
    Hp, PRE, x = np.concatenate(Hp), np.concatenate(PRE), np.concatenate([X[:, t] for t in range(10)])
    wi_m, wh_m = [], []
    for a, j in enumerate(MEM):
        on = PRE[:, j] > 0
        tgt = PRE[:, j] - Hp[:, MEM] @ Wm[a]
        F = np.c_[Hp[:, OUT], x]
        sol = np.linalg.lstsq(F[on], tgt[on], rcond=None)[0]
        wh_m.append(sol[:-1]); wi_m.append(sol[-1])
    S = h @ WO.T; S = S - S.mean(1, keepdims=True)
    sol = np.linalg.lstsq(h, S, rcond=None)[0]
    wo = (sol.T - sol.T.mean(0, keepdims=True)) + WO.mean(0, keepdims=True)
    return np.r_[wi_m, np.ravel(wh_m), wo.ravel()]


_C0 = th.tensor(init_constants())
_cw = th.tensor(_rows(_block(_C0[:3], _C0[3:5], _C0[5:].reshape(8, 8)).numpy()))


def weights(C):
    out = _block(C[:3], C[3:5], C[5:].reshape(8, 8)).reshape(-1)
    return th.cat([out, _cw.to(out)])

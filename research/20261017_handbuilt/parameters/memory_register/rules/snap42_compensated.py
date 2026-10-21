"""Check rule (step 5d, not a candidate): the memory block with ARC's eigenvectors and shrink factors, angles set to
1, 2, 3 x 42 deg (0 and 180 deg kept). With COMPENSATE (default on), the readout and the memory's incoming weights
from outside the memory are the step-5 states-fit weights for this block (least squares on ARC's states, set (b);
step5_weights/snap42_b.npz). They depend on no constant of the rule: they are derived from the block by that
analysis procedure. Constants: 64 basis numbers, 3 shrink factors, 2 real eigenvalues, read off ARC's weights
exactly (fit() returns them unchanged, so the basis does not bend to undo the snap).
COMPENSATE=0: block only. COMPENSATE=arc: the same rows set to ARC's own weights (a leak check, see README step 5d).
STRUCTURE_BITS: 4 (as the eigen re-expression) + 6 for the hand-set 42 deg + 3 for the multiples 1, 2, 3."""
import os
from pathlib import Path
import numpy as np, torch as th
from algzoo.core import example
MEM = [0, 3, 10, 11, 12, 13, 14, 15]
OUT = [i for i in range(16) if i not in MEM]
COMPENSATE = os.environ.get("COMPENSATE", "1") != "0"
ARC_ROWS = os.environ.get("COMPENSATE") == "arc"   # leak check: the same rows set to ARC's own weights
_, WH, _ = example()
_ev, _V = np.linalg.eig(WH[np.ix_(MEM, MEM)])
_pairs = sorted([k for k in range(8) if _ev[k].imag > 1e-9], key=lambda k: np.angle(_ev[k]))   # 38, 84, 127
_reals = [k for k in range(8) if abs(_ev[k].imag) < 1e-9]
_B = np.concatenate([np.stack([_V[:, k].real, _V[:, k].imag], 1) for k in _pairs] + [_V[:, [k]].real for k in _reals], 1)
ANG = th.tensor(np.radians(42.0) * np.array([1, 2, 3]))
BLOCK = [f"{i}->{j}" for j in MEM for i in MEM]
COMP = ([f"input->{j}" for j in MEM] + [f"{i}->{j}" for j in MEM for i in OUT] +
        [f"{i}->out{k}" for k in range(10) for i in range(16)]) if COMPENSATE else []
COVERED = BLOCK + COMP
if COMPENSATE:
    _d = np.load(Path(__file__).resolve().parent.parent / "step5_weights" / "snap42_b.npz")
    wi_, wh_, wo_ = example() if ARC_ROWS else (_d["states_wi"], _d["states_wh"], _d["states_wo"])
    _cw = th.tensor(np.r_[[wi_[j] for j in MEM], [wh_[j, i] for j in MEM for i in OUT],
                          [wo_[k, i] for k in range(10) for i in range(16)]])


def init_constants():
    return np.concatenate([[abs(_ev[k]) for k in _pairs], [_ev[k].real for k in _reals], _B.ravel()])


def fit(C0, target, F):
    return C0


KINDS = np.array(["shrink"] * 3 + ["real"] * 2 + ["basis"] * 64)
STRUCTURE_BITS = 13


def weights(C):
    r, re, B = C[:3], C[3:5], C[5:].reshape(8, 8)
    blocks = [th.stack([th.stack([r[k] * th.cos(ANG[k]), r[k] * th.sin(ANG[k])]),
                        th.stack([-r[k] * th.sin(ANG[k]), r[k] * th.cos(ANG[k])])]) for k in range(3)]
    W = B @ th.block_diag(*blocks, th.diag(re)) @ th.linalg.inv(B)
    out = W.reshape(-1)
    return th.cat([out, _cw.to(out)]) if COMPENSATE else out

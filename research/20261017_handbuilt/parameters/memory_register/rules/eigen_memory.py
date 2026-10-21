"""Example rule: the memory block (memory -> memory, 64 weights) as V diag(lambda) V^-1 in real form.
Constants: the 8 eigenvalues as (shrink, angle) for 3 rotating pairs plus 2 real ones, and the real basis
(8 x 8, columns = real and imaginary parts of the eigenvectors). Read off ARC's weights exactly; this is a
re-expression, a check that the scorer prices a rule that explains nothing as a loss."""
import numpy as np, torch as th
from algzoo.core import example
MEM = [0, 3, 10, 11, 12, 13, 14, 15]
COVERED = [f"{i}->{j}" for j in MEM for i in MEM]
_, WH, _ = example()
_ev, _V = np.linalg.eig(WH[np.ix_(MEM, MEM)])
_pairs = [k for k in range(8) if _ev[k].imag > 1e-9]
_reals = [k for k in range(8) if abs(_ev[k].imag) < 1e-9]
_B = np.concatenate([np.stack([_V[:, k].real, _V[:, k].imag], 1) for k in _pairs] + [_V[:, [k]].real for k in _reals], 1)


def init_constants():
    eig = [abs(_ev[k]) for k in _pairs] + [np.angle(_ev[k]) for k in _pairs] + [_ev[k].real for k in _reals]
    return np.concatenate([eig, _B.ravel()])


KINDS = np.array(["shrink"] * 3 + ["angle"] * 3 + ["real"] * 2 + ["basis"] * 64)
STRUCTURE_BITS = 4


def weights(C):
    r, a, re, B = C[:3], C[3:6], C[6:8], C[8:].reshape(8, 8)
    blocks = [th.stack([th.stack([r[k] * th.cos(a[k]), r[k] * th.sin(a[k])]), th.stack([-r[k] * th.sin(a[k]), r[k] * th.cos(a[k])])]) for k in range(3)]
    Lam = th.block_diag(*blocks, th.diag(re))
    W = B @ Lam @ th.linalg.inv(B)  # W[j, i] from i to j, rows/cols in MEM order
    return W.reshape(-1)

"""Noise-floor rule (step 0 of the loop; also the readout shift relation as a rule). The memory's readout row for
position k+1 is row k pushed one step through the memory block: W_oh[k+1, MEM] = g_k W_oh[k, MEM] W_mm, for k = 0..6,
with ARC's memory block W_mm (not covered, so stated anyway). Covers the readout's memory columns at positions 0-7.
Constants: row 0's 8 memory weights (kind "row0") and 7 gains g_k (kind "gain"). With SHARED_GAIN=1 one gain for all.
Fit: least squares on ARC's states (an analysis fit, list seed FIT_SEED, default 101): minimise the mean squared
difference between the rule's and ARC's memory readout of ARC's final memory states, over 2^16 lists. No answers used.
STRUCTURE_BITS: 1 (the form: each row is the previous one pushed through the block) + 3 (where the chain ends: 7 of
positions 1-9) + 1 (gains per step or shared) = 5."""
import os
import numpy as np, torch as th
from algzoo.core import example
MEM = [0, 3, 10, 11, 12, 13, 14, 15]
POS = list(range(8))
COVERED = [f"{i}->out{k}" for k in POS for i in MEM]
SHARED = os.environ.get("SHARED_GAIN", "0") == "1"
SEED = int(os.environ.get("FIT_SEED", "101"))
WI, WH, WO = example()
_B = th.tensor(WH[np.ix_(MEM, MEM)])
NG = 1 if SHARED else len(POS) - 1
KINDS = np.array(["row0"] * 8 + ["gain"] * NG)
STRUCTURE_BITS = 5


def init_constants():
    return np.r_[WO[0, MEM], np.ones(NG)]


def weights(C):
    rows = [C[:8]]
    for k in range(len(POS) - 1):
        rows.append(C[8 + (0 if SHARED else k)] * (rows[-1] @ _B))
    return th.cat(rows)


def fit(C0, target, F):
    X = np.random.default_rng(SEED).standard_normal((2 ** 16, 10))
    h = np.zeros((len(X), 16))
    for t in range(10):
        h = np.maximum(h @ WH.T + X[:, t, None] * WI, 0)
    Hm = th.tensor(h[:, MEM])
    G = Hm.T @ Hm / len(Hm)                      # second moment of ARC's final memory state
    tg = th.tensor(target).reshape(len(POS), 8)
    C = th.tensor(C0, dtype=th.float64, requires_grad=True)
    opt = th.optim.Adam([C], lr=1e-3)
    for _ in range(4000):
        opt.zero_grad(); d = weights(C).reshape(len(POS), 8) - tg
        L = th.einsum("ki,ij,kj->", d, G, d); L.backward(); opt.step()
    return C.detach().numpy()

"""Check rule for leak_check_chain.py (not a candidate). Covers the memory columns of readout rows 1-3.
CHAIN=forward (default): row k = g x ARC's row k-1 (memory columns) pushed through ARC's memory block; ORDER out1,
out2, out3. A valid chain: must pass. CHAIN=reverse: the same rule declared in the order out3, out2, out1, so rows
depend on rows after them: must fail. CHAIN=self: each row = g x ARC's own row: must fail."""
import os
import numpy as np, torch as th
from algzoo.core import example
MEM = [0, 3, 10, 11, 12, 13, 14, 15]
MODE = os.environ.get("CHAIN", "forward")
WI, WH, WO = example()
COVERED = [f"{i}->out{k}" for k in (1, 2, 3) for i in MEM]
ORDER = ["out3", "out2", "out1"] if MODE == "reverse" else ["out1", "out2", "out3"]
KINDS = np.array(["gain"])
STRUCTURE_BITS = 3
_src = th.tensor(np.stack([WO[k, MEM] for k in (1, 2, 3)]) if MODE == "self" else
                 np.stack([WO[k - 1, MEM] @ WH[np.ix_(MEM, MEM)] for k in (1, 2, 3)]))


def init_constants():
    return np.ones(1)


def weights(C):
    return (C[0] * _src).reshape(-1)

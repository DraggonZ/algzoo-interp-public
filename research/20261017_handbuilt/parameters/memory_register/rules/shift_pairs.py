"""Fair votes as a one-step shift of the readout, each covered row pushed from ARC's uncovered neighbour row.

Mechanism. A number at position k is seen at the end through the memory and hiding neurons after 9 - k steps; a
number at position k + 1 after one step fewer. If both are read at the same height (fair votes, ../README.md
experiment 12), position k + 1's readout equals position k's readout pushed one step through those neurons'
recurrent weights:

    W_oh[k + 1, MH] = g * W_oh[k, MH] @ W_hh[MH, MH],   MH = memory 0, 3, 10-15 and hiding 1, 6, 7, 8.

Rule S-5 (rules/shift_readout.py) chained this from position 0, so its errors compounded (relative error 0.40), and
it used the memory columns only. Here each covered row k + 1 is pushed from ARC's row k, which is NOT covered (stated
anyway, free), so each row carries only its own one-step error, and the hiding columns are included (they carry the
largest readout weights). Covered positions and their sources: PAIRS (env, target:source, default 1:0, 2:0, 4:3, 5:3, 7:6;
the best of three tried, r1b-2 and r1b-3 in LEDGER.md); a target
further than one step from its source is pushed that many steps. Sources are uncovered.
Default ON=1: the push runs through the neurons that are on in the most common on/off pattern (all but detector 4 and
gates 5, 9) and also covers the store's and the keep gate's readout columns, which follow the same push.
One constant: a shared gain g (1 if the votes are exactly fair).

STRUCTURE_BITS:
  the form (row pushed one step through the block, forward direction): 1 + 1
  which columns: memory and hiding groups out of the four groups (2^4 choices): 4. With ON=1: the push runs
    through the neurons on in the most common on/off pattern (detector 4, gates 5, 9 off; 2 bits, the pattern of
    the blindness rule) and covers the memory, hiding, store and keep-gate columns (6 bits: six units, store,
    detector, keep gate, late gate, hiding, memory, each in or out)
  which target positions: log2 C(9, k) of positions 1-9 (chosen after looking at push errors on ARC)
  which source positions (each target is pushed from the nearest source below it): log2 C(9, m), 0 if every
    target is one step above its source
  the form "one shared gain": 1
"""
import math
import os
import numpy as np
import torch as th
from algzoo.core import example

WI, WH, WO = example()
MH = [0, 3, 10, 11, 12, 13, 14, 15, 1, 6, 7, 8]
# PAIRS: target:source; the target is the source pushed (target - source) steps
PAIRS = [tuple(int(a) for a in p.split(":")) for p in os.environ.get("PAIRS", "1:0,2:0,4:3,5:3,7:6").split(",")]
TARGETS = [t for t, _ in PAIRS]
assert not set(TARGETS) & {s for _, s in PAIRS} and all(t > s for t, s in PAIRS), "sources must be uncovered"
# ON=1: the push runs through the neurons on in the most common pattern (all but 4, 5, 9) and also covers the
# store's and keep gate's columns
ON = os.environ.get("ON", "1") == "1"
KINDS = np.array(["gain"])
_SOURCES = sorted({s for _, s in PAIRS})
_ONE_STEP = all(t - s == 1 for t, s in PAIRS)   # then the sources follow from the targets
STRUCTURE_BITS = (2 + (6 + 2 if ON else 4) + round(math.log2(math.comb(9, len(TARGETS))))
                  + (0 if _ONE_STEP else round(math.log2(math.comb(9, len(_SOURCES))))) + 1)
SRC = [i for i in range(16) if i not in (4, 5, 9)] if ON else MH
TGT = MH + [2, 5] if ON else MH
_B = WH[np.ix_(SRC, SRC)]
COVERED = [f"{i}->out{k}" for k in TARGETS for i in TGT]
_P = th.tensor(np.concatenate([WO[s, SRC] @ np.linalg.matrix_power(_B, t - s - 1) @ WH[np.ix_(SRC, TGT)]
                               for t, s in PAIRS]), dtype=th.float64)


def init_constants():
    return np.array([1.0])


def weights(C):
    return C[0] * _P


def counterparts(C):
    return [("g", float(C[0]), None)]

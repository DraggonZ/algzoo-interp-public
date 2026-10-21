"""Blindness to the arriving number, as a rule for the memory's weights from hiding neuron 7.

Mechanism. The hiding neurons 1, 6, 7, 8 have input weights near -11, so after a number x arrives they carry about
-11 x (plus their reading of the largest so far). A memory neuron j reads them through W[j, 1], W[j, 6], W[j, 7],
W[j, 8]. Blindness: the sum of those reads, weighted by the hiding neurons' input weights, is zero, so the number that
just arrived does not reach the memory on the next step. For neurons 10-13 and 3 the cancellation runs through the
whole state in its most common on/off pattern (detector 4 and gates 5, 9 off): the memory neurons' own small input
weights take part (FINDINGS section 2; ../README.md experiments 9-11). Solved for the weight from neuron 7:

    W[j, 7] = -g * sum_{i in S_j, i != 7} W[j, i] WI[i] / WI[7],
    S_j = {1, 6, 7, 8} (class "hiding") or all neurons except 4, 5, 9 (class "state").

Every weight on the right is ARC's and not covered (stated anyway), so the covered weights depend on ARC only
through them. One constant: a shared gain g (1 if blindness is exact); the scorer needs at least one constant.

STRUCTURE_BITS:
  which rows are covered: log2 C(8, k) of the eight memory rows (ROWS env, default all eight)
  class per covered row ("hiding" or "state"): 1 bit each. The classes were chosen from earlier per-row costs
    (../README.md experiment 10), so they are priced, not derived.
  which of the four hiding weights is solved for: 2 bits (one choice for all rows)
  the on/off pattern for class "state" (the most common one, measured): 2 bits
  the form "times one gain": 1 bit
"""
import math
import os
import numpy as np
import torch as th
from algzoo.core import example

WI, WH, WO = example()
STATE_ROWS = {3, 10, 11, 12, 13}
ROWS = [int(r) for r in os.environ.get("ROWS", "0,3,10,11,12,13,14,15").split(",")]
SOLVE = 7
HID = [1, 6, 7, 8]
ALL_ON = [i for i in range(16) if i not in (4, 5, 9)]
COVERED = [f"{SOLVE}->{j}" for j in ROWS]
KINDS = np.array(["gain"])
STRUCTURE_BITS = round(math.log2(math.comb(8, len(ROWS)))) + len(ROWS) + 2 + 2 + 1


def _blind_value(j):
    S = ALL_ON if j in STATE_ROWS else HID
    rest = sum(WH[j, i] * WI[i] for i in S if i != SOLVE)
    return -rest / WI[SOLVE]


_B = th.tensor([_blind_value(j) for j in ROWS], dtype=th.float64)


def init_constants():
    return np.array([1.0])


def weights(C):
    return C[0] * _B


def counterparts(C):
    return [("g", float(C[0]), None)]

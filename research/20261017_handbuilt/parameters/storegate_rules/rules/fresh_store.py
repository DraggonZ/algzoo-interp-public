"""r6b: the detector reads the fresh store.

Mechanism. The store (neuron 2) holds about 17.6 x the largest; the detector (neuron 4) fires when an arriving number
beats it. The detector does not compare the number with the store's current value but with the value the store is
about to take: both neurons read the same state h_t, and the detector's recurrent drive is -c times the store's own
recurrent pre-activation,
    W[4, i] = -c * W[2, i]   for all 16 sources i (store, detector, gates, hiding, memory),
so d_{t+1} = ReLU(k x_t - c * s_{t+1}) with k = input->4 (uncovered). Row 2 is uncovered (ARC's, stated anyway).
The store's own fade (2->2) and its accumulation of the detector (4->2) thereby fix the detector's self-weight
(4->4 = -c * 4->2) and its read of the store (2->4 = -c * 2->2).

One constant: c (ARC's least-squares value 0.552).

ROWS env var (default "4") is for looks only.

STRUCTURE_BITS (4): which row is read (1 of 3 other rows in the store/detector/gate set): 2; form (minus a multiple
of that row, all 16 columns): 1; input weight left out: 1.
"""
import numpy as np
import torch as th
from algzoo.core import example

WI, WH, WO = example()
STRUCTURE_BITS = 4
KINDS = np.array(["ratio"])
COVERED = [f"{i}->4" for i in range(16)]
_ROW2 = th.tensor(WH[2], dtype=th.float64)


def init_constants():
    return np.array([0.55])


def weights(C):
    return -C[0] * _ROW2


def counterparts(C):
    return [("c", float(C[0]), None)]

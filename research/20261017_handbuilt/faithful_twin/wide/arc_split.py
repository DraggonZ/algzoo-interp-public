"""Calibration candidate for score_wide.py: ARC with hiding neuron 7 and memory neuron 11 each split into two exact
copies (same incoming weights, outgoing weights halved), 18 neurons. MAP: ARC slot i = neuron i, averaged with the copy for
slots 7 and 11, so the mapped activity equals ARC's exactly. Must score as ARC (12 of 12, same transplants)."""
import numpy as np
from algzoo.core import example
SPLIT = {7: 16, 11: 17}
def weights():
    wi, wh, wo = [np.asarray(a, float) for a in example()]; n = 18
    Wi = np.zeros(n); Wh = np.zeros((n, n)); Wo = np.zeros((10, n))
    src = list(range(16)) + list(SPLIT)            # wider neuron k copies ARC neuron src[k]
    for k in range(n):
        Wi[k] = wi[src[k]]; Wo[:, k] = wo[:, src[k]] / (2 if src[k] in SPLIT else 1)
        for l in range(n): Wh[k, l] = wh[src[k], src[l]] / (2 if src[l] in SPLIT else 1)
    return Wi, Wh, Wo
MAP = np.zeros((16, 18)); MAP[np.arange(16), np.arange(16)] = 1
for i, k in SPLIT.items(): MAP[i, i] = MAP[i, k] = 0.5

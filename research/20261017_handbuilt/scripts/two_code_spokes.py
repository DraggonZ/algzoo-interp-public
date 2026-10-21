"""Spokes of ARC's readout and of the two order-specific readouts (two_code_readouts.pt) in the dial plane.
A spoke is a position's readout row restricted to the twelve dial neurons, centred over positions, projected on the
dial axes (viz/model.json). Readouts are compared after scaling each to the same overall size (argmax is scale-free)."""
import json

import numpy as np
import torch as th

from algzoo.core import example

m = json.load(open("viz/model.json")); idx = m["dial_neurons"]; axes = np.array(m["dial_axes"])
wo = example()[2]; R = th.load("two_code_readouts.pt")
rd = {"ARC": wo, "second before largest": R["before"].numpy(), "second after largest": R["after"].numpy()}


def spokes(W):
    W = W / np.linalg.norm(W)
    S = W[:, idx]; S = S - S.mean(0)
    P = S @ axes.T
    return np.degrees(np.arctan2(P[:, 1], P[:, 0])), np.hypot(P[:, 0], P[:, 1]), np.linalg.norm(S, axis=1)


base = spokes(wo)
for name, W in rd.items():
    ang, ln, full = spokes(W)
    d = (ang - base[0] + 180) % 360 - 180
    print(f"{name:22s} angle change vs ARC per position: " + " ".join(f"{x:+5.0f}" for x in d) +
          " | spoke length / ARC: " + " ".join(f"{a/b:4.2f}" for a, b in zip(ln, base[1])))

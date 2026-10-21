"""Hypothesis: robustness is set by the ratio of the hiding group's wave (the part that carries position and votes) to
its level (the shared part that only keeps the neurons positive). Noise scales with a neuron's whole value; the signal is
the wave. Per list and step, split the group's state h (4 or 3+1 neurons) into its projection on the all-ones direction
(level) and the rest (wave); report the median ratio |wave| / |level| on live steps, and how often some but not all of the
group are off. Trained network: cancelling neurons 1, 6, 7, 8. Twin (trained scale): clock 4-6 and flip 3."""
import numpy as np
from algzoo.handbuilt_v18 import build_v18
import simlib

X = simlib.lists(73, 20_000); age = simlib.task_vars(X)["age"]; live = age >= 1
for name, P, G in (("trained", (simlib.WI, simlib.WH, simlib.WO), [1, 6, 7, 8]),
                   ("twin, trained scale", tuple(np.asarray(p, float) for p in build_v18(g=10, a=10, w=10, lam=10.0, c=-9.0, eps=0.1).params), [3, 4, 5, 6]),
                   ("twin, g 10000", tuple(np.asarray(p, float) for p in build_v18().params), [3, 4, 5, 6])):
    P = (np.asarray(P[0], float).reshape(-1), np.asarray(P[1], float), np.asarray(P[2], float))
    _, Pre, H = simlib.run(X, params=P); h = H[:, :, G]; u = np.ones(len(G)) / np.sqrt(len(G))
    lev = h @ u; wave = np.linalg.norm(h - lev[..., None] * u, axis=-1)
    on = h.sum(-1) > 1e-9; m = live & on
    off = (Pre[:, :, G] <= 0).sum(-1); part = ((off > 0) & (off < len(G)))[live].mean()
    print(f"{name:22s}: |wave| / |level| median {np.median(wave[m] / np.maximum(lev[m], 1e-12)):.3f} (90th percentile {np.percentile(wave[m] / np.maximum(lev[m], 1e-12), 90):.3f}); partly off on {part:.3f} of live steps")

"""v19 (kappa 0.5) with a fading clock: its turning part shrinks by rho per step (the trained timer shrinks about 6%
per step: account item 7, 0.84-0.95). The flip neuron's turning part fades the same way. Nothing else changes, so the
memory's copies of the wave no longer match what the readout subtracts: an imperfect hiding whose leftover grows with
the hidden amount and with age. rho from the trained range, not tuned to the scores."""
import numpy as np
from common import *
from algzoo.handbuilt_v19 import build_v19
import score
def build(rho=1.0, kappa=0.5):
    net = build_v19(kappa=kappa); wi, wh, wo = [p.copy() for p in net.params]
    C = [4, 5, 6]; F = 3
    wh[np.ix_(C, C)] *= rho
    wh[F, F] = -rho
    for k in C: wh[F, k] += (rho - 1) / 3      # the level restore follows the self-weight (-rho F + rho level)
    return wi, wh, wo
if __name__ == "__main__":
    models = {f"v19 fade rho {r}": [T(build(r))] for r in (1.0, 0.97, 0.94, 0.9, 0.85)}
    score.report(models)

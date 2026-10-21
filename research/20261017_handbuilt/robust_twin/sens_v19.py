"""Per-neuron noise sensitivity (sensitivity.py) for v19 at trained scale, kappa 0 and 0.5."""
import numpy as np
from algzoo.handbuilt_v19 import build_v19
import simlib
from sensitivity import sens, NAMES18
X = simlib.lists(71, 20_000)
for kappa in (0.0, 0.5):
    s, typ = sens(build_v19(g=10, a=10, w=10, lam=10.0, c=-9.0, eps=0.1, kappa=kappa).params, X)
    print(f"kappa {kappa}: " + "; ".join(f"{NAMES18[n]} {s[n]:.2f}" for n in np.argsort(-s)[:8]))

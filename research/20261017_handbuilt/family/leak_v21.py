"""leak_match.py on v21 (28 neurons, built for robustness: real margins). Separates 'the leak is wrong' from 'v19 is too
fragile to carry any leak'."""
import numpy as np, torch
from common import *
from algzoo.handbuilt_v21 import build_v21
from crosstalk import measure
from leak_match import effects
import score
if __name__ == "__main__":
    K = measure(); I = np.eye(10); wi, wh, wo = build_v21()
    X = near_lists(30000, seed=4); L, S, Tq = top3(X); models = {}
    for s in (0.0, 0.05, 0.1, 0.2, 0.3):
        W = T([wi, wh, (I + s * (K - I)) @ wo]); e = effects(W, X, L, S, Tq)
        print(f"v21 s {s}: |effect| per unit: positive before L {e[0]:.3f}, after L {e[1]:.3f}, negative {e[2]:.3f}  (ARC 0.103, 0.039, 0.007)")
        models[f"v21 + leak s {s}"] = [W]
    score.report(models)

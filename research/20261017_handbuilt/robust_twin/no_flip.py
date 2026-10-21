"""How much does v18's flip neuron (beta, the alternating term that sharpens the revealed wave) buy? beta 0.5 against 0,
at v18's gains and at trained scale. 200,000 lists."""
import numpy as np
from algzoo.handbuilt_v19 import build_v19
import simlib
X = simlib.lists(76, 200_000); y = X.argsort(1)[:, -2]
for lab, kw in (("v18 gains", {}), ("trained scale", dict(g=10, a=10, w=10, lam=10.0, c=-9.0, eps=0.1))):
    for beta in (0.5, 0.0):
        P = tuple(np.asarray(p, float) for p in build_v19(beta=beta, **kw).params)
        print(f"{lab:13s} beta {beta}: accuracy {(simlib.run(X, params=P)[0].argmax(1) == y).mean():.4f}", flush=True)

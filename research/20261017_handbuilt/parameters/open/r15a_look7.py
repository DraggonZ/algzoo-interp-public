"""r15a look 7: centred leftover per row for r14a and for the contender push (top 3, fresh arrival)."""
import numpy as np
from r15a_var import *

np.set_printoptions(precision=2, suppress=True, linewidth=200)
rank = (-XS).argsort(1).argsort(1)
w3 = np.stack([(rank[:, k] < 3).astype(float) for k in PUSH_ROWS])
row0 = np.tile(-WO[0] / 9, (8, 1))
rng = np.random.default_rng(1101)
Ac = WO - WO.mean(0)
for lab, v in [("r14a", Var(push(step(H, XS.min(1))), row0, 8)),
               ("top3 fresh", Var(push(step(H, rng.standard_normal(N)), w3)))]:
    net, d = XC.price(v)
    P = d["wf"][272:].reshape(10, 16); E = (P - P.mean(0)) - Ac
    print(lab, "v2c %+.1f" % net)
    print(" row rel err", np.linalg.norm(E, axis=1) / np.linalg.norm(Ac, axis=1))
    print(" col rms err", np.sqrt((E[1:] ** 2).mean(0)))
    print(E)

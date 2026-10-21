"""Memory modes for the page: projections onto the three rotating pairs (positive-angle eigenvector of each),
and each memory neuron's mark (direction of the mode state that pushes it up most)."""
import json, sys, numpy as np
sys.path.insert(0, "/data/output/src/algzoo/src")
from algzoo.core import example
wi, wh, wo = (np.asarray(a, float) for a in example())
MEM = [0, 3, 10, 11, 12, 13, 14, 15]
A = wh[np.ix_(MEM, MEM)]
ev, V = np.linalg.eig(A); U = np.linalg.inv(V)
modes = []
for k in sorted([k for k in range(8) if 1e-6 < np.angle(ev[k]) < 3.1], key=lambda k: np.angle(ev[k])):
    c = wh[np.ix_(MEM, MEM)] @ V[:, k]          # input to each memory neuron per unit of mode state: 2 Re(c_j z)
    marks = {int(MEM[i]): float(np.degrees(-np.angle(c[i]))) for i in range(8)}
    modes.append(dict(angle=float(np.degrees(np.angle(ev[k]))), shrink=float(abs(ev[k])),
                      ur=U[k].real.tolist(), ui=U[k].imag.tolist(), marks=marks))
w = json.load(open("arc_weights.json"))
w["modes"] = modes; w["mem"] = MEM
json.dump(w, open("page_data.json", "w"))
for m in modes: print(round(m["angle"], 1), round(m["shrink"], 3), {j: round(a) for j, a in m["marks"].items()})

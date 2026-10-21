"""r6b look 1: which stated vectors line up with the target rows' sub-blocks?
For each target neuron j in 2, 4, 5, 9 and each column group (memory, hiding, all recurrent), the cosine of
W[j, group] with every other row W[k, group] and with W[group, k] (a column), top 4."""
import numpy as np
from algzoo.core import example

WI, WH, WO = example()
mem = [0, 3, 10, 11, 12, 13, 14, 15]; hid = [1, 6, 7, 8]
groups = {"memory": mem, "hiding": hid, "all": list(range(16))}
cos = lambda a, b: float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
np.set_printoptions(precision=3, suppress=True, linewidth=160)
for j in [2, 4, 5, 9]:
    for gname, g in groups.items():
        v = WH[j, g]
        cands = [(cos(v, WH[k, g]), f"row {k}") for k in range(16) if k != j]
        cands += [(cos(v, WH[g, k]), f"col {k}") for k in range(16)]
        cands += [(cos(v, WI[g]), "input")] + [(cos(v, WO[k, g]), f"out{k}") for k in range(10)]
        cands.sort(key=lambda c: -abs(c[0]))
        print(j, gname, " ".join(f"{n}:{c:+.2f}" for c, n in cands[:5]))
# blindness numbers: hiding read dotted with hiding input weights, relative
for j in [2, 4, 5, 9]:
    v = WH[j, hid]
    print(j, "hiding read", v, "cos with input weights", round(cos(v, WI[hid]), 3))
# detector vs store row
r2, r4 = WH[2], WH[4]
c = (r2 @ r4) / (r2 @ r2)
print("row4 = c row2, c", round(c, 3), "resid", (r4 - c * r2).round(3))

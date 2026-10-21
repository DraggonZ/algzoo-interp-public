"""Look: off-rate of each memory neuron by (age of the largest, age of the second-largest), all steps pooled (seed 32).
Rows: age of the largest (-1 = none positive); columns: age of the second (-1 = none). '.' = fewer than 200 samples.
Ages of the second below the largest's mean it came after the largest (kept or not); above, it came before (an
overtaken record or an earlier number)."""
import numpy as np
from swlib import *

Xf = lists(FIT_SEED, 200_000); _, Pf, _ = run(Xf); off = (Pf[:, :, MEM] <= 0).reshape(-1, 8)
agf, _ = order_keys(Xf); a1 = agf[:, :, 0].ravel(); a2 = agf[:, :, 1].ravel()
for j, n in enumerate(MEM):
    print(f"neuron {n}: off-rate (%) by age of largest (rows) and of second (columns -1..9)")
    print("        " + "".join(f"{c:5d}" for c in range(-1, 10)))
    for r in range(-1, 10):
        cells = []
        for c in range(-1, 10):
            m = (a1 == r) & (a2 == c)
            cells.append(f"{100 * off[m, j].mean():5.0f}" if m.sum() >= 200 else "    .")
        print(f"   {r:4d} " + "".join(cells))

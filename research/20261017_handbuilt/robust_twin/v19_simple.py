"""v19 on constructed lists (background -3 so small numbers vote nothing): (a) a lone 0.6 at position p; (b) 0.6 at p,
then 1.0 at q (the 0.6 is overtaken, so it should win with vote (1 - c) 0.6); (c) 0.6 at p, 1.0 at q, 0.5 later.
Scores per position for kappa 0 and 0.5 (v18 gains; scores / (1 - c))."""
import numpy as np
from algzoo.handbuilt_v19 import build_v19
import simlib
np.set_printoptions(precision=3, suppress=True, linewidth=150)
cases = []
for p, q in ((2, 5), (4, 6), (6, 7)):
    x = np.full(10, -3.0); x[p] = 0.6; x[q] = 1.0; cases.append((f"0.6 at {p}, 1.0 at {q}", x))
x = np.full(10, -3.0); x[3] = 0.6; cases.append(("lone 0.6 at 3", x))
x = np.full(10, -3.0); x[2] = 0.3; x[4] = 0.6; x[7] = 1.0; cases.append(("0.3 at 2, 0.6 at 4, 1.0 at 7", x))
for kappa in (0.0, 0.5):
    P = tuple(np.asarray(p, float) for p in build_v19(kappa=kappa).params)
    for lab, x in cases:
        print(f"kappa {kappa}, {lab:28s}:", simlib.run(x[None], params=P)[0][0] / 0.25)

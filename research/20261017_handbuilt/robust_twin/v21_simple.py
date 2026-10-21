"""v20 on constructed lists (background -3): scores per position / A0, against what the rules say they should be."""
import numpy as np
from algzoo.handbuilt_v21 import build_v21 as build_v20
import simlib
np.set_printoptions(precision=3, suppress=True, linewidth=160)
P = build_v20()
cases = []
for p, q in ((2, 5), (4, 6), (6, 7), (1, 8), (3, 9), (7, 8), (8, 9)):
    x = np.full(10, -3.0); x[p] = 0.6; x[q] = 1.0; cases.append((f"0.6 at {p}, 1.0 at {q}", x))
x = np.full(10, -3.0); x[3] = 0.6; cases.append(("lone 0.6 at 3", x))
x = np.full(10, -3.0); x[2] = 0.3; x[4] = 0.6; x[7] = 1.0; cases.append(("0.3 at 2, 0.6 at 4, 1.0 at 7", x))
x = np.full(10, -3.0); x[1] = 1.0; x[4] = 0.6; x[6] = 0.4; cases.append(("1.0 at 1, 0.6 at 4, 0.4 at 6", x))
x = np.full(10, -3.0); x[1] = 0.5; x[2] = 0.7; x[5] = 1.0; cases.append(("0.5 at 1, 0.7 at 2, 1.0 at 5", x))
for lab, x in cases:
    print(f"{lab:30s}:", simlib.run(x[None], params=P)[0][0] / 10.0)
X = simlib.lists(88, 200_000); y = X.argsort(1)[:, -2]
print("accuracy on random lists:", np.mean(simlib.run(X, params=P)[0].argmax(1) == y))

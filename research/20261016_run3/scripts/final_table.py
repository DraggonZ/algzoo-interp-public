"""Final neuron values in the two-spike case: fix the largest's position, move the second across positions."""
import sys

import numpy as np

from algzoo.core import example

r = 0.7
w_hi, w_hh, w_oh = example()
C = [1, 2, 4, 6, 7, 8]
order = C + [i for i in range(16) if i not in C]


def run(x):
    h = np.zeros(16)
    for t in range(10):
        h = np.maximum(0, w_hh @ h + x[t] * w_hi)
    return h


for p1 in [int(a) for a in sys.argv[1:]] or [0, 4, 9]:
    print(f"\nlargest = 1 at position {p1}; second = {r} at the position in each column")
    cols = [p for p in range(10) if p != p1]
    Hf = {p2: run(np.where(np.arange(10) == p1, 1.0, np.where(np.arange(10) == p2, r, 0.0))) for p2 in cols}
    print("       " + " ".join(f"p2={p:<4d}" for p in cols))
    for i in order:
        print(f"{'C' if i in C else ' '} n{i:<3d}" + " ".join(f"{Hf[p][i]:7.2f}" for p in cols))
    print("answer " + " ".join(f"{int(np.argmax(w_oh @ Hf[p])):7d}" for p in cols))

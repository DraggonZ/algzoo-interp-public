"""Why does neuron 5's keep threshold vary (0.6-0.85 of the second-largest so far)? Constructed lists: largest M at
position 0, the second so far S = 1.0 at position pS, the new number x at position px > pS, -0.3 elsewhere (also 0).
Threshold = smallest x at which neuron 5 fires (> 0.05) one step after px, as a fraction of S. Vary the age of S at
x's arrival (px - pS), how long ago the largest arrived, and M."""
import numpy as np

from algzoo.core import example

wi, wh, wo = example()
xs = np.linspace(0.2, 1.6, 1401)


def n5_after(X, px):
    h = np.zeros((len(X), 16)); out = None
    for t in range(10):
        h = np.maximum(0, h @ wh.T + X[:, t:t + 1] * wi)
        if t == px + 1:
            out = h[:, 5]
    return out


for bg in (-0.3, 0.0):
    print(f"filler {bg}")
    for M in (1.5, 2.5):
        rows = []
        for pS in (1, 2, 3, 4, 5, 6):
            row = []
            for px in range(pS + 1, 9):
                X = np.full((len(xs), 10), bg); X[:, 0] = M; X[:, pS] = 1.0; X[:, px] = xs
                n5 = n5_after(X, px); on = np.where(n5 > 0.05)[0]
                row.append(f"{xs[on[0]]:.2f}" if len(on) else " -  ")
            rows.append(f"  S at {pS}: x at {pS+1}..8 -> " + " ".join(row))
        print(f" M = {M} (largest at 0)"); print("\n".join(rows))

"""An internal test of forgetting, which the output-based resampling test could not provide (it passed on v18 too).
Nudge one positive number at position s (1-7) by +0.01 and measure how much the memory neurons' final activity moves per
unit (norm of the change / 0.01). Compare numbers the account says are forgotten (below 0.75 x the second-largest
before them) with kept numbers (at or above it, but not the largest or second-largest of the whole list).
Trained network: memory = 0, 3, 10-15; v18: its eight delay-line slots. If the network forgets, the ratio
forgotten / kept should be near 0. v18 keeps every positive number, so its ratio should be near 1."""
import numpy as np
import simlib
from algzoo.handbuilt_v18 import build_v18
X = simlib.lists(303, 200_000); B = len(X)
nets = {"trained": ((simlib.WI, simlib.WH, simlib.WO), [0, 3, 10, 11, 12, 13, 14, 15]),
        "v18": (tuple(np.asarray(p, float) for p in build_v18().params), list(range(8, 16)))}
for s in range(1, 8):
    P = np.c_[np.zeros((B, 2)), X[:, :s]]; m2 = np.sort(P, 1)[:, -2]
    top2 = np.argsort(X, 1)[:, -2:]; is_top = (top2 == s).any(1)
    x = X[:, s]; forgotten = (x > 0) & (x < 0.75 * m2) & ~is_top; kept = (x > 0) & (x >= 0.75 * m2) & ~is_top & (x < np.max(P, 1))
    X2 = X.copy(); X2[:, s] += 0.01
    for name, (Pm, mem) in nets.items():
        H = simlib.run(X, params=Pm)[2][:, 9, mem]; H2 = simlib.run(X2, params=Pm)[2][:, 9, mem]
        g = np.linalg.norm(H2 - H, axis=1) / 0.01
        gf, gk = np.median(g[forgotten]), np.median(g[kept])
        print(f"position {s} {name:8s}: memory moves per unit, median: forgotten number {gf:.3f} (n={forgotten.sum()}), kept number {gk:.3f} (n={kept.sum()}), ratio {gf / gk:.3f}")

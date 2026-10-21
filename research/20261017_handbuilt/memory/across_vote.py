"""Question: does a kept vote spill forward (onto positions after its number) in the trained networks? Role-free.
Lists whose second-largest is at s (2-7), after the largest, and > 0.3; nudge x_s; per-unit final-score change, mean.
Printed per network: the vote at s (mean over s), the mean over s of the largest change at s+1..8, the mean over s of
the largest change at s-3..s-1, and position 9. Networks and lists as across_tail.py (400,000 lists, seed 14)."""
import numpy as np
from across_tail import models, X, scores, eps
o = X.argsort(1)
print("network | vote at s | largest later (s+1..8) | largest earlier (s-3..s-1) | pos 9")
for name, w in models.items():
    P = [np.asarray(p, float) for p in w]; P[0] = P[0].reshape(-1); v, f, b, p9 = [], [], [], []
    for s in range(2, 8):
        Xs = X[(o[:, -2] == s) & (o[:, -1] < s) & (X[:, s] > 0.3)]; dX = np.zeros_like(Xs); dX[:, s] = eps
        d = ((scores(P, Xs + dX) - scores(P, Xs)) / eps).mean(0)
        v.append(d[s]); f.append(d[s + 1:9].max()); b.append(d[max(0, s - 3):s].max()); p9.append(d[9])
    print(f"{name:7s} | {np.mean(v):6.1f} | {np.mean(f):6.1f} | {np.mean(b):6.1f} | {np.mean(p9):6.1f}")

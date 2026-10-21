"""Which kept numbers do the cancelling neurons hold? Nudge one kept number (not the list's largest or second-largest)
at position s by 0.01 and measure how far the four cancelling neurons' final activity moves per unit, split by whether s
comes after the current largest (the largest of the whole list) or before it. Memory neurons as reference. 400,000
lists; s = 3..6."""
import numpy as np
import simlib
X = simlib.lists(909, 400_000); B = len(X); L, _, H = simlib.run(X)
lar = X.argmax(1); top = np.argsort(X, 1)[:, -2:]
print(f"{'s':>2s} {'where':>14s} {'n':>7s} {'cancellers':>10s} {'memory':>7s}")
for s in range(3, 7):
    P = np.c_[np.zeros((B, 2)), X[:, :s]]; m2 = np.sort(P, 1)[:, -2]; x = X[:, s]
    kept = (x > 0) & (x >= 0.75 * m2) & ~(top == s).any(1) & (x < P.max(1))
    X2 = X.copy(); X2[:, s] += 0.01; H2 = simlib.run(X2)[2]
    gc = np.linalg.norm(H2[:, 9, [1, 6, 7, 8]] - H[:, 9, [1, 6, 7, 8]], axis=1) / 0.01
    gm = np.linalg.norm(H2[:, 9, [0, 3, 10, 11, 12, 13, 14, 15]] - H[:, 9, [0, 3, 10, 11, 12, 13, 14, 15]], axis=1) / 0.01
    for lab, m in (("after largest", kept & (lar < s)), ("before largest", kept & (lar > s))):
        print(f"{s:2d} {lab:>14s} {m.sum():7d} {np.median(gc[m]):10.2f} {np.median(gm[m]):7.2f}")

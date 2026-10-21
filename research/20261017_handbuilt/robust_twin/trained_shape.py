"""The trained network's hiding pattern: at the final step, each position's score split into the cancelling neurons'
part (w_o[p, C] h_C) and the rest, as a function of the offset d = p - r from the largest's position r, in units of the
largest's value m1. Lists where the largest is at position 1-5 (so offsets -1..+4 exist), median over lists. Centred:
each part minus its mean over the ten positions (a constant does not change the answer)."""
import numpy as np
import simlib

X = simlib.lists(85, 200_000); C = [1, 6, 7, 8]; R = [n for n in range(16) if n not in C]
_, _, H = simlib.run(X); h = H[:, -1]
canc = h[:, C] @ simlib.WO[:, C].T; rest = h[:, R] @ simlib.WO[:, R].T
canc -= canc.mean(1, keepdims=True); rest -= rest.mean(1, keepdims=True)
r = X.argmax(1); m1 = X.max(1); ok = (r >= 1) & (r <= 5) & (m1 > 0.5)
for lab, part in (("cancelling neurons", canc), ("rest (memory, store, gates)", rest), ("total", canc + rest)):
    row = []
    for d in range(-1, 5):
        v = part[ok, r[ok] + d] / m1[ok]; row.append(f"{d:+d}: {np.median(v):+6.1f}")
    print(f"{lab:28s} per unit of m1: " + "  ".join(row))
# the same for positions holding a kept positive number vs a non-positive number at offset +2

"""Question: what does one unit of each writer put into each memory mode, and where on the score line does each mode's
share land? Weights only, memory all on and linear (no switching), no feedback to the other neurons.

A unit of activity of writer s at step t-1 writes B[:, s] into the memory at step t. Ten minus t steps later its
share in mode family f is read at position k as
    g_f(k) = sum over the modes j in f of Re( W_oh[k, mem] v_j  lambda_j^(9-t)  (V^-1 B[:, s])_j ).
The five g_f add up to the writer's whole final-score pattern W_oh[k, mem] A^(9-t) B[:, s].
Printed for a write at step 5 (positions 0-9; position 5 = the write step) and the sum for a write at step 7.
Also each mode's readout size by position (how strongly the readout sees a unit of that mode, with its phase) is in
modes.log."""
import numpy as np
from memlib import A, B, WIM, RM, OTH, NAMES

np.set_printoptions(precision=1, suppress=True, linewidth=180)
lam, V = np.linalg.eig(A); Vinv = np.linalg.inv(V)
fams = {}
for j, l in enumerate(lam): fams.setdefault(round(abs(np.degrees(np.angle(l)))), []).append(j)
writers = {NAMES[s]: B[:, i] for i, s in enumerate(OTH)}; writers["input"] = WIM


def split(u, t):
    a = Vinv @ u; out = {}
    for f, js in sorted(fams.items()):
        out[f] = sum(np.real(RM @ V[:, j] * lam[j] ** (9 - t) * a[j]) for j in js)
    return out


for name, u in writers.items():
    print(f"== {name}: one unit written at step 5 -> final scores at positions 0-9, by mode family")
    parts = split(u, 5)
    for f, g in parts.items(): print(f"   {f:4d} deg  ", " ".join(f"{x:6.1f}" for x in g))
    print(f"   sum       ", " ".join(f"{x:6.1f}" for x in sum(parts.values())))
    print(f"   sum, written at step 7", " ".join(f"{x:6.1f}" for x in sum(split(u, 7).values())))

print("\n== where each writer's unit write lands, by write step t (memory all on, linear): position of the largest positive and")
print("   of the largest negative final score among positions 0-7 (written as offset from t), value; and the largest value at positions after t (0-7 only)")
for name, u in writers.items():
    cells = []
    for t in range(3, 10):
        g = sum(split(u, t).values())[:8]; kp, kn = int(np.argmax(g)), int(np.argmin(g))
        fwd = np.abs(g[t + 1:8]).max() if t + 1 < 8 else 0.0
        cells.append(f"t{t}: +{g[kp]:.1f}@{kp - t:+d} {g[kn]:.1f}@{kn - t:+d} fwd {fwd:.1f}")
    print(f"   {name:12s} " + " | ".join(cells))

print("\n== share of each writer's final-score pattern (positions 0-7, write at step 5) in each mode family: sum over positions of")
print("   the family's part times the whole pattern, divided by the whole pattern's sum of squares (shares add to 1); and peak / typical other position")
print("   writer       " + " ".join(f"{f:>6d}" for f in sorted(fams)) + "   peak/others")
for name, u in writers.items():
    parts = split(u, 5); g = sum(parts.values())[:8]
    sh = [np.dot(p[:8], g) / np.dot(g, g) for p in parts.values()]
    k = int(np.argmax(np.abs(g))); others = np.sqrt(np.mean(np.delete(g, k) ** 2))
    print(f"   {name:12s} " + " ".join(f"{s:6.2f}" for s in sh) + f"   {abs(g[k]) / others:5.1f} (peak {g[k]:+.1f} at {k})")

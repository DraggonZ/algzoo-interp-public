"""Geometry of the cancelling neurons' 4-vector (neurons 1, 6, 7, 8). Share of variance in its top principal
directions; which subsets are off on partly switched steps; how partial switching depends on the timer's age (steps
since the largest arrived) and on the new number's size relative to the largest."""
from collections import Counter
import numpy as np
import simlib
from simlib import task_vars

X = simlib.lists(2828, 100_000); _, PRE, H = simlib.run(X); V = task_vars(X); C = [1, 6, 7, 8]
c = H[:, :, C].reshape(-1, 4); act = (c > 0).any(1)
U, S, Vt = np.linalg.svd(c[act] - c[act].mean(0), full_matrices=False)
print("variance share of principal directions:", np.round(S ** 2 / (S ** 2).sum(), 3))
on = (H[:, :, C] > 0); k = on.sum(-1); part = (k > 0) & (k < 4)
cnt = Counter(tuple(int(v) for v in row) for row in on[part])
print("partly switched steps: which of (1, 6, 7, 8) are on, most common:", [(p, round(n / part.sum(), 3)) for p, n in cnt.most_common(6)])
age = V["age"]; r = np.where(V["m1"] > 0, V["x"] / np.where(V["m1"] > 0, V["m1"], 1), np.nan)
print("partial switching by timer age (steps since the largest), positive numbers below the largest (ratio 0-1):")
for a in range(1, 9):
    m = (age == a) & (r > 0) & (r < 1)
    if m.sum() > 500:
        print(f"  age {a}: {part[m].mean():.2f} of {m.sum()} steps; median ratio at which it happens {np.nanmedian(r[m & part]):.2f}")
print("pre-activations of the four at partly switched steps, median (on / off):")
for i, n in enumerate(C):
    print(f"  neuron {n}: on {np.median(PRE[:, :, n][part & on[:, :, i]]):6.2f}, off {np.median(PRE[:, :, n][part & ~on[:, :, i]]):6.2f}")

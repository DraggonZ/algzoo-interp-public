"""Neuron 15's off-rate by step (rows) and by the largest's age (columns: -1 none yet, 0 record now, 1, 2, ...), and
the same table for neuron 0, to look for a rule."""
import numpy as np
from simlib import lists, run, task_vars
X = lists(202); _, P, _ = run(X); V = task_vars(X); age = V["age"].astype(int)
for n in (15, 0):
    off = P[:, :, n] < 0
    print(f"neuron {n}: off-rate, rows = step 0-9, columns = age -1..9 ('.' = no cases)")
    for t in range(10):
        row = []
        for a in range(-1, 10):
            m = age[:, t] == a
            row.append(f"{off[m, t].mean():4.2f}" if m.sum() > 200 else "   .")
        print(f"  step {t}: " + " ".join(row))

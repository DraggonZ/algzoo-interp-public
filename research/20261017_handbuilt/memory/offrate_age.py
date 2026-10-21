"""Question: is each memory neuron's switching a function of the age of the largest (the timer's phase)? Off-rate of each
memory neuron by exact age (0-9; -1 = no positive yet) and by step. 100,000 lists, seed 11."""
import numpy as np
from memlib import MEM, run, task, lists

X = lists(11, 100_000); V = task(X); age = V["age"]; S0, P0, H0 = run(X)
for n in MEM:
    off = P0[:, :, n] <= 0
    print(f"neuron {n}: off-rate, rows = step 0-9, columns = age -1..9 ('.' = no cases)")
    for t in range(10):
        cells = []
        for a in range(-1, 10):
            m = age[:, t] == a
            cells.append(f"{off[m, t].mean():5.2f}" if m.sum() > 50 else "    .")
        print(f"  step {t}: " + " ".join(cells))

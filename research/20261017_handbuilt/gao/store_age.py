"""Does the store (neurons 2, 4) carry when the largest arrived? Among steps with no new largest, regress each neuron
on the largest so far, the input and the second-largest, then show the leftover by the largest's age."""
import numpy as np
from simlib import lists, run, task_vars
X = lists(202); _, P, H = run(X); V = task_vars(X)
sel = (V["newmax"] == 0) & (V["age"] >= 1) & (V["m1"] > 0.3)
for n in (2, 4, 5):
    y = H[:, :, n][sel]; A = np.c_[V["m1in"][sel], V["x"][sel], V["m2in"][sel]]
    co = np.linalg.lstsq(A, y, rcond=None)[0]; r = y - A @ co; age = V["age"][sel]; m1 = V["m1in"][sel]
    print(f"neuron {n}: coefficients largest {co[0]:.2f}, input {co[1]:.2f}, second {co[2]:.2f}; leftover / largest by age:")
    print("   " + "  ".join(f"{a}: {np.mean(r[age == a] / m1[age == a]):+.3f}" for a in range(1, 10)))

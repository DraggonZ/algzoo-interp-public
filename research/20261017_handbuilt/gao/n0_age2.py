"""What does neuron 0's switch-off two steps after a new largest remove? Regress its pre-activation at that step on
the new largest, the old largest (previous record), the second-largest and the input; and show its size."""
import numpy as np
from simlib import lists, run, task_vars
from sim_rich import record_vars
X = lists(202); _, P, _ = run(X); V = task_vars(X); prev_rec, after, _ = record_vars(X)
sel = V["age"] == 2; y = P[:, :, 0][sel]
cols = {"largest": V["m1in"][sel], "previous record": prev_rec[sel], "second": V["m2in"][sel], "input": V["x"][sel], "input one step ago": np.c_[np.zeros(len(X)), X[:, :-1]][sel]}
A = np.c_[tuple(cols.values())]; co = np.linalg.lstsq(A, y, rcond=None)[0]; r2 = 1 - ((A @ co - y) ** 2).mean() / y.var()
print(f"pre-activation of neuron 0 two steps after a new largest: mean {y.mean():.2f}, 10th-90th percentile {np.percentile(y, 10):.2f} to {np.percentile(y, 90):.2f}")
print("fit: " + ", ".join(f"{k} {c:+.2f}" for k, c in zip(cols, co)) + f"; R2 {r2:.3f}")

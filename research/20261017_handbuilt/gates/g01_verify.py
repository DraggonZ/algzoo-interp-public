"""Check the known claims about the two gates before building on them (200,000 random lists, seed 1).
Neuron 5 (keep gate): on one step after x_t when x_t >= about 0.75 x the second-largest so far; every record passes.
Neuron 9 (late gate): on one step after a new largest, rarely otherwise."""
import numpy as np
from gatelib import lists, run, task_vars, answer

X = lists(1); L, P, H = run(X); V = task_vars(X)
print(f"baseline accuracy {100 * (L.argmax(1) == answer(X)).mean():.2f}%")
on5 = H[:, 1:, 5] > 0; on9 = H[:, 1:, 9] > 0             # value one step after x_t, t = 0..8
x, m1, m2, new = (V[k][:, :9] for k in ("x", "m1", "m2", "new"))
new = new.astype(bool); pos = x > 0
print("\nneuron 5, one step after x_t (t = 0..8)")
print(f"  x_t a record:          on {on5[new].mean():.4f} of {new.sum()} steps")
nr = ~new & (m2 > 0.2)
r = x / np.where(m2 > 0, m2, 1)
bins = [-9, 0, 0.3, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 1.0, 1.2, 99]
print("  not a record, second-largest so far m2 > 0.2: share on by x_t / m2")
for lo, hi in zip(bins[:-1], bins[1:]):
    s = nr & (r >= lo) & (r < hi)
    print(f"    [{lo:5.2f},{hi:5.2f}): on {on5[s].mean():.3f}  ({s.sum()} steps)")
# 50% point per list-group: for each (age of largest, age of second) cell, threshold where share crosses 0.5
print("  50% point of x_t / m2 by step t (non-records, m2 > 0.2)")
for t in range(2, 9):
    s = nr[:, t]; rr = r[s, t]; oo = on5[s, t]
    grid = np.linspace(0.3, 1.2, 91); sh = np.array([oo[(rr >= g - 0.025) & (rr < g + 0.025)].mean() for g in grid])
    k = np.argmax(sh >= 0.5)
    print(f"    t={t}: {grid[k]:.2f}")
print("\nneuron 9, one step after x_t")
print(f"  x_t a record: on {on9[new].mean():.4f}; mean value {H[:, 1:, 9][new].mean():.2f}")
print(f"  otherwise:    on {on9[~new].mean():.4f}; mean value {H[:, 1:, 9][~new].mean():.3f}")
print(f"  record with a positive previous largest (m1 > 0): on {on9[new & (m1 > 0)].mean():.4f}")
print(f"  record with no positive number before (m1 = 0):   on {on9[new & (m1 == 0)].mean():.4f}")
# neuron 9 at the record's own step and two steps after
print(f"  neuron 9 at the record's own step: on {(H[:, :9, 9] > 0)[new].mean():.4f}")
print(f"  neuron 9 at step 0 (no step after anything): on {(H[:, 0, 9] > 0).mean():.4f}")
print(f"  neuron 5 at step 0: on {(H[:, 0, 5] > 0).mean():.4f}")

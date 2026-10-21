"""In the trained network, when a kept number just below the largest (within 5%) arrives at least two steps after the
record and silences the cancelling neurons (1, 6, 7, 8) fully or partly, do they recover on the following steps, and in
which direction? Sum of the four and the angle between their state and the state they would have had (the same list with
that number replaced by -1, a number that does not silence them), at the arrival step and 1, 2, 3 steps later."""
import numpy as np
import simlib

X = simlib.lists(82, 300_000); B = len(X); C = [1, 6, 7, 8]
_, Pre, H = simlib.run(X)
cm = np.maximum.accumulate(X, 1); prev = np.c_[np.zeros((B, 1)), cm[:, :-1]]
age = simlib.task_vars(X)["age"]
rows = {k: [] for k in range(4)}; rows_base = {k: [] for k in range(4)}; angs = {k: [] for k in range(4)}
for t in range(3, 7):
    x = X[:, t]; m = (x < prev[:, t]) & (x > 0.95 * prev[:, t]) & (age[:, t] >= 2) & ((Pre[:, t][:, C] <= 0).sum(1) > 0)
    idx = np.flatnonzero(m)[:5000]
    if len(idx) == 0: continue
    Xb = X[idx].copy(); Xb[:, t] = -1.0; _, _, Hb = simlib.run(Xb)
    for k in range(4):
        a = H[idx, t + k][:, C]; b = Hb[:, t + k][:, C]
        rows[k] += list(a.sum(1)); rows_base[k] += list(b.sum(1))
        cos = (a * b).sum(1) / np.maximum(np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-9)
        angs[k] += list(np.degrees(np.arccos(np.clip(cos, -1, 1))))
for k in range(4):
    print(f"{k} step(s) after the arrival: sum of the four {np.median(rows[k]):6.2f} (without the silencing number {np.median(rows_base[k]):6.2f}); angle to the unsilenced state {np.median(angs[k]):5.1f} degrees ({len(rows[k])} lists)")

"""Does the trained cancellers' wave shrink when the current number approaches the largest? Pre-activation of neurons
1, 6, 7, 8 split into level (along all-ones) and wave (orthogonal), at live steps (record at least 2 steps earlier),
binned by (m1 - x_t) / m1. Also the same split of the pre-activation's input part alone and of its recurrent part."""
import numpy as np
import simlib
X = simlib.lists(87, 200_000); B = len(X); C = [1, 6, 7, 8]; u = np.ones(4) / 2
_, Pre, H = simlib.run(X); age = simlib.task_vars(X)["age"]
cm = np.maximum.accumulate(X, 1); prev = np.c_[np.zeros((B, 1)), cm[:, :-1]]
rel = (prev - X) / np.maximum(prev, 1e-9)
pre = Pre[:, :, C]; lev = pre @ u; wave = np.linalg.norm(pre - lev[..., None] * u, axis=-1)
rec = np.einsum("btn,cn->btc", np.concatenate([np.zeros((B, 1, 16)), H[:, :-1]], 1), simlib.WH[C]); rw = np.linalg.norm(rec - (rec @ u)[..., None] * u, axis=-1)
live = (age >= 2) & (prev > 0.5)
for lo, hi in ((-1, 0), (0, 0.05), (0.05, 0.2), (0.2, 0.5), (0.5, 1), (1, 3), (3, 99)):
    m = live & (rel >= lo) & (rel < hi)
    print(f"(m1 - x)/m1 in [{lo}, {hi}): {m.sum():7d} steps; level {np.median(lev[m]):6.1f}; wave of the pre-activation {np.median(wave[m]):5.2f}; wave of the recurrent part alone {np.median(rw[m]):5.2f}")

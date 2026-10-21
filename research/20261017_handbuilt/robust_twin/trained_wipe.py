"""How does the trained network silence its cancelling neurons (1, 6, 7, 8) when a record barely beats the previous
largest? At record steps (age of the previous record >= 2), binned by relative jump d / x: share of steps with all four
off, with some off, the median sum of the four (residual) against the median sum one step earlier, and the median
pre-activation of each of the four."""
import numpy as np
import simlib

X = simlib.lists(80, 200_000); B = len(X); C = [1, 6, 7, 8]
_, Pre, H = simlib.run(X)
cm = np.maximum.accumulate(X, 1); prev = np.c_[np.zeros((B, 1)), cm[:, :-1]]
bins = [0, 0.01, 0.02, 0.05, 0.1, 0.3, 1.01]
for lo, hi in zip(bins[:-1], bins[1:]):
    rows = []
    for t in range(2, 10):
        x = X[:, t]; p = prev[:, t]; m = (x > p) & (p > 0) & ((x - p) / x >= lo) & ((x - p) / x < hi)
        if m.sum() == 0: continue
        rows.append((Pre[m, t][:, C], H[m, t - 1][:, C]))
    pre = np.concatenate([r[0] for r in rows]); before = np.concatenate([r[1] for r in rows])
    off = (pre <= 0).sum(1)
    print(f"jump {lo:.2f}-{hi:.2f} ({len(pre)} record steps): all four off {np.mean(off == 4):.3f}, some off {np.mean((off > 0) & (off < 4)):.3f}; "
          f"residual sum {np.median(np.maximum(pre, 0).sum(1)):.2f} against {np.median(before.sum(1)):.2f} one step earlier; median pre-activations " + " ".join(f"{v:.1f}" for v in np.median(pre, 0)))

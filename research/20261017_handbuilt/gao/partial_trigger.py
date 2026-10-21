"""What makes the cancelling neurons switch off partly (some of 1, 6, 7, 8 on, some off)? Per step, the share of
steps with partial switching, binned by the new number's ratio to the largest so far before it (positive largest).
Also how often lists whose second-largest is within 10% of the largest have partial switching at some step."""
import numpy as np
import simlib
from simlib import task_vars

X = simlib.lists(2525, 200_000); _, _, H = simlib.run(X); V = task_vars(X)
on = (H[:, :, [1, 6, 7, 8]] > 0).sum(-1); partial = (on > 0) & (on < 4)
r = np.where(V["m1"] > 0, V["x"] / np.where(V["m1"] > 0, V["m1"], 1), np.nan)
print(f"partial switching on {partial[:, 1:].mean():.3f} of steps 1-9")
for a, b in ((-9, 0), (0, 0.5), (0.5, 0.8), (0.8, 0.9), (0.9, 0.95), (0.95, 1.0), (1.0, 1.05), (1.05, 1.2), (1.2, 99)):
    m = (r >= a) & (r < b); m[:, 0] = False
    print(f"new number / largest so far {a:5.2f}-{b:5.2f}: {m.sum() / m[:, 1:].size:.3f} of steps; partial switching on {partial[m].mean():.3f} of them")
srt = np.sort(X, 1); close = srt[:, -2] / srt[:, -1] > 0.9
anyp = partial.any(1)
print(f"lists whose second-largest is within 10% of the largest: {close.mean():.3f}; partial switching somewhere in {anyp[close].mean():.2f} of them, against {anyp[~close].mean():.2f} otherwise")

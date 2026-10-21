"""Tie offsets from one linearized step. For each of the 720 placements of the top
three (largest 1.6, second 1.0, background -0.46; tie_bias_stability.py), the exact offset comes from bisection on the
third's value. Newton step from an exact tie: offset ~ -(s3 - s2) / (d(s3 - s2)/dx3) at x3 = 1.0 (derivative by a
1e-4 finite difference). Also: how much of s3 - s2 at the tie is cancellation (sum of |pushes| on it over |s3 - s2|)."""
import io, contextlib
import numpy as np
import torch as th
with contextlib.redirect_stdout(io.StringIO()):
    from tie_sensitivity_across import bias, pl, N, r
from algzoo.core import example, forward
import simlib

P = example(); B0 = bias(P)
base = np.full((N, 10), -0.46); base[r, pl[:, 0]] = 1.6; base[r, pl[:, 1]] = 1.0
def gap(x3):
    x = base.copy(); x[r, pl[:, 2]] = x3; s = forward(P, th.tensor(x, device="cuda")).cpu().numpy()
    return s[r, pl[:, 2]] - s[r, pl[:, 1]], x
g0, x0 = gap(1.0); g1, _ = gap(1.0 + 1e-4); slope = (g1 - g0) / 1e-4
newton = -g0 / slope
print(f"exact offsets: typical |offset| {np.median(np.abs(B0)):.4f}; Newton from the tie: correlation {np.corrcoef(B0, newton)[0, 1]:.3f}, median |difference| {np.median(np.abs(B0 - newton)):.4f}")
_, _, Hh = simlib.run(x0); d = simlib.WO[pl[:, 2]] - simlib.WO[pl[:, 1]]; push = d * Hh[:, -1]
print(f"at the exact tie: |s3 - s2| median {np.median(np.abs(g0)):.2f}; sum of |pushes| on it median {np.median(np.abs(push).sum(1)):.0f}; ratio median {np.median(np.abs(push).sum(1) / np.abs(g0)):.0f}")
print(f"slope d(s3 - s2)/dx3 median {np.median(slope):.1f} per unit (so an offset of 0.016 is a score gap of about {0.016 * np.median(slope):.2f})")
from scipy.stats import spearmanr
close = np.abs(B0 - newton) < 0.002
print(f"robust: rank correlation {spearmanr(B0, newton)[0]:.3f}; within 0.002 of the exact offset on {close.mean():.3f} of placements; "
      f"outliers' median slope {np.median(slope[~close]):.1f} against {np.median(slope[close]):.1f}")

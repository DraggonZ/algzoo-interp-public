"""How much does a forgotten (filler) number still move a close call? At the 720 filler ties (g = 1.6), the exact
per-input slope of the margin (bias_anatomy) for each filler position, compared with the slope for the third number.
A filler's effect per unit relative to a kept vote predicts how much real fillers shift the flip point."""
import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from bias_anatomy import anatomy  # noqa: E402

from algzoo.core import example  # noqa: E402

pl = np.array(list(itertools.permutations(range(10), 3))); r = np.arange(len(pl))
m, _, _, _, c = anatomy(example(), 1.6)
c3 = c[r, pl[:, 2]]
mask = np.ones((len(pl), 10), bool); mask[r, pl[:, 0]] = mask[r, pl[:, 1]] = mask[r, pl[:, 2]] = False
rel = np.abs(c[mask].reshape(len(pl), 7)) / np.abs(c3)[:, None]
print(f"third's slope median {np.median(np.abs(c3)):.1f}; filler slope / third's slope: median {np.median(rel):.4f}, 90th percentile {np.quantile(rel, 0.9):.4f}")
# Predicted spread of the flip point from real fillers: fillers ~ the 7 smallest of 10 normals, scaled by the second.
rng = np.random.default_rng(0); Z = np.sort(rng.standard_normal((20000, 10)), 1)
f = Z[:, :7] / Z[:, 8:9]                                     # fillers relative to the second (positive second)
f = f[Z[:, 8] > 0.5]                                         # typical close calls: second well above zero
cf = (c[mask].reshape(len(pl), 7) / c3[:, None])             # per placement, fillers' shift of the margin per unit, in third-units
sh = np.concatenate([f[:2000] @ cf[i] - np.mean(f[:2000] @ cf[i]) for i in range(len(pl))])
print(f"flip-point shift from real fillers around its mean (fraction of the second): median |shift| {np.median(np.abs(sh)):.4f}, "
      f"90th percentile {np.quantile(np.abs(sh), 0.9):.4f}; typical tie bias |m / c3| median {np.median(np.abs(m / c3)):.4f}")

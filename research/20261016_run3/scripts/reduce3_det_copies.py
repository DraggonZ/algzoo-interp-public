"""Diagnostic: the frozen sampling-free three-input estimate (reduce3_det.py) on the 96 noisy copies, per noise level."""
import sys

import numpy as np

sys.path.insert(0, "/data/output/research/algzoo/20261016_run3/scripts")
from reduce3_det import estimate  # noqa: E402

from algzoo.bench import load_bench  # noqa: E402

names, acc, models = load_bench(("mult",))
sig = np.array([float(str(n).split("_")[0][4:]) for n in names])
est = np.array([estimate(p)[0] for p in models]); acc = np.asarray(acc)
for s in (0.005, 0.01, 0.02):
    k = sig == s
    ok = k & (acc >= 0.6)
    e = 100 * (est[k] - acc[k]); e2 = 100 * (est[ok] - acc[ok])
    print(f"sigma {s}: {k.sum()} copies, RMSE {np.sqrt(np.mean(e**2)):.2f} points (bias {e.mean():+.2f}); "
          f"copies at 0.6 or above ({ok.sum()}): RMSE {np.sqrt(np.mean(e2**2)):.2f}; accuracy range {acc[k].min():.2f}-{acc[k].max():.2f}")
e = 100 * (est - acc)
print(f"all: RMSE {np.sqrt(np.mean(e**2)):.2f}; correlation {np.corrcoef(est, acc)[0, 1]:.3f}")

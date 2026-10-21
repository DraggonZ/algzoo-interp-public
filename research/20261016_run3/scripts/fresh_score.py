"""Score the frozen sampling-free three-input estimate once on the 50 unseen models (true accuracies computed first)."""
import sys

import numpy as np

sys.path.insert(0, "/data/output/research/algzoo/20261016_run3/scripts")
from reduce3_det import estimate  # noqa: E402

from algzoo.core import zoo  # noqa: E402

R = np.load("/data/output/research/algzoo/20261016_run3/reports/fresh_truth.npy")
rows = []
for e, s, a in R:
    est, ops, _ = estimate(zoo(16, 10, int(s), n_seqs=2 ** int(e)))
    rows.append((e, s, a, est))
    print(f"n_seqs 2^{int(e)} seed {int(s)}: true {a:.4f} estimate {est:.4f} error {100*(est-a):+.1f} points", flush=True)
X = np.array(rows)
err = X[:, 3] - X[:, 2]
print(f"cost {ops/2880:.0f} samples' worth per model")
for lo, hi, lab in [(0.0, 0.6, "accuracy below 0.6"), (0.6, 0.85, "0.6-0.85"), (0.85, 1.0, "above 0.85")]:
    k = (X[:, 2] >= lo) & (X[:, 2] < hi)
    print(f"{lab} ({k.sum()} models): RMSE {100*np.sqrt(np.mean(err[k]**2)):.2f} points, bias {100*np.mean(err[k]):+.2f}")
print(f"all 50: RMSE {100*np.sqrt(np.mean(err**2)):.2f} points; correlation {np.corrcoef(X[:,2], X[:,3])[0,1]:.3f}")
np.save("/data/output/research/algzoo/20261016_run3/reports/fresh_score.npy", X)

"""The frozen estimate against sampling at equal compute, with shared prefixes counted (reduce3_trie.shared_cost).
Estimates are unchanged from reduce3_det.log and fresh_score.log; only the cost is recounted. Sampling's expected squared
error at n runs is p(1 - p) / n, with p the true accuracy."""
import sys

import numpy as np

sys.path.insert(0, "/data/output/research/algzoo/20261016_run3/scripts")
from reduce3_trie import shared_cost  # noqa: E402

from algzoo.core import example, zoo  # noqa: E402

truth = {"example": 0.9532, 0: 0.9101, 1: 0.8804, 2: 0.9485, 3: 0.9320, 4: 0.8898}
rows = []
for name, a in truth.items():
    est, _, ops, *_ = shared_cost(example() if name == "example" else zoo(16, 10, name))
    n = ops / 2880; rows.append((est - a, a * (1 - a) / n, n))
R = np.array(rows)
print(f"six trained models: cost {R[:, 2].min():.0f}-{R[:, 2].max():.0f} samples' worth; estimate RMSE {100 * np.sqrt(np.mean(R[:, 0] ** 2)):.2f} points, "
      f"sampling at the same cost {100 * np.sqrt(np.mean(R[:, 1])):.2f}; ratio of squared errors {np.mean(R[:, 0] ** 2) / np.mean(R[:, 1]):.2f}")
F = np.load("/data/output/research/algzoo/20261016_run3/reports/fresh_score.npy")
rows = []
for e, s, a, est in F:
    _, _, ops, *_ = shared_cost(zoo(16, 10, int(s), n_seqs=2 ** int(e)))
    n = ops / 2880; rows.append((a, est - a, a * (1 - a) / n, n))
R = np.array(rows)
for lo, hi, lab in [(0.0, 0.6, "accuracy below 0.6"), (0.6, 0.85, "0.6-0.85"), (0.85, 1.0, "above 0.85"), (0, 1, "all 50")]:
    k = (R[:, 0] >= lo) & (R[:, 0] < hi)
    print(f"unseen checkpoints, {lab} ({k.sum()}): cost {R[k, 3].min():.0f}-{R[k, 3].max():.0f}; estimate RMSE {100 * np.sqrt(np.mean(R[k, 1] ** 2)):.2f} points, "
          f"sampling {100 * np.sqrt(np.mean(R[k, 2])):.2f}; ratio of squared errors {np.mean(R[k, 1] ** 2) / np.mean(R[k, 2]):.2f}")

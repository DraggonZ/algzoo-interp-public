"""Step 2 of the new-seed check: the frozen estimate, scored once on the 16 new models, against sampling at equal compute
(shared-prefix cost; integral at 30 operations per point and placement, 6 shown alongside). Pre-registered in the run
history, steps 74 and 76. Also reports each model's sampled three-input reduction as a diagnostic."""
import sys

import numpy as np

sys.path.insert(0, "/data/output/research/algzoo/20261016_run3/scripts")
from reduce3_trie import shared_cost  # noqa: E402

from algzoo.core import load  # noqa: E402

T = np.load("/data/output/research/algzoo/20261016_run3/reports/new_truth.npy")
rows = []
for seed, a in T:
    est, ops_old, ops, *_ = shared_cost(load(f"/data/output/research/algzoo/20261016_run3/models/2nd_argmax_16_10_1073741824_{int(seed)}.pth"))
    n6 = ops / 2880; n15 = n6 + 720 * 64 * 24 / 2880   # second column: 30 ops per point
    rows.append((seed, a, est, n6, n15))
    print(f"seed {int(seed)}: true {a:.4f} estimate {est:.4f} error {100 * (est - a):+.2f} points; cost {n15:.0f} samples' worth at 30 ops per point ({n6:.0f} at 6)", flush=True)
R = np.array(rows); np.save("/data/output/research/algzoo/20261016_run3/reports/new_score.npy", R)
err = R[:, 2] - R[:, 1]
for lab, k in [("all 16", np.ones(len(R), bool)), ("accuracy 0.85 or above", R[:, 1] >= 0.85)]:
    if k.sum() == 0:
        continue
    for j, conv in [(4, "30 ops per point (primary)"), (3, "6 ops per point")]:
        samp = R[k, 1] * (1 - R[k, 1]) / R[k, j]
        rng = np.random.default_rng(0); idx = np.flatnonzero(k)
        boot = [np.mean(err[b] ** 2) / np.mean(R[b, 1] * (1 - R[b, 1]) / R[b, j]) for b in (rng.choice(idx, len(idx)) for _ in range(10000))]
        print(f"{lab} ({k.sum()}), {conv}: estimate RMSE {100 * np.sqrt(np.mean(err[k] ** 2)):.2f} points, sampling {100 * np.sqrt(np.mean(samp)):.2f}; "
              f"ratio of squared errors {np.mean(err[k] ** 2) / np.mean(samp):.2f} (bootstrap 90% range {np.percentile(boot, 5):.2f}-{np.percentile(boot, 95):.2f}); bias {100 * err[k].mean():+.2f}")
const = R[:, 1].mean()
print(f"a constant equal to the mean accuracy would have RMSE {100 * np.sqrt(np.mean((R[:, 1] - const) ** 2)):.2f} points (uses the true mean, for scale only)")

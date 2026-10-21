"""Pre-registered check (run history step 100): the simple formula, unchanged, on 16 fresh models (seeds 200-215).
Step 1 computes and saves true accuracies (2^26 inputs each); step 2 scores the formula once."""
import io
import contextlib
import sys

import numpy as np

from algzoo.core import accuracy, load

sys.path.insert(0, "/data/output/research/algzoo/20261016_run3/scripts")
D = "/data/output/research/algzoo/20261016_run3"
seeds = list(range(200, 216))
truth = []
for s in seeds:
    a, _ = accuracy(load(f"{D}/models/2nd_argmax_16_10_1073741824_{s}.pth"), n_samples=2 ** 26, seed=s)
    truth.append(a); print(f"seed {s}: true accuracy {a:.4f}", flush=True)
np.save(f"{D}/reports/fresh200_truth.npy", np.c_[seeds, truth])
with contextlib.redirect_stdout(io.StringIO()):
    import simple_estimator as SE        # its own 22-model printout is suppressed
rows = []
for s, a in zip(seeds, truth):
    p = load(f"{D}/models/2nd_argmax_16_10_1073741824_{s}.pth")
    w23, w12, cost = SE.widths(p)
    est = 1 - (8 / 9) * SE.P1 - 2.54 * w23 - 1.54 * w12
    rows.append((a, est, cost)); print(f"seed {s}: estimate {est:.4f} ({100 * (est - a):+.2f} points); cost {cost:.0f} runs' worth", flush=True)
R = np.array(rows); e = R[:, 1] - R[:, 0]; samp = R[:, 0] * (1 - R[:, 0]) / R[:, 2]
rng = np.random.default_rng(0)
boot = [np.mean(e[b] ** 2) / np.mean(samp[b]) for b in (rng.integers(0, 16, 16) for _ in range(10000))]
print(f"16 fresh models: RMSE {100 * np.sqrt(np.mean(e ** 2)):.2f} points, bias {100 * e.mean():+.2f}; sampling at the same cost {100 * np.sqrt(np.mean(samp)):.2f}; "
      f"ratio of squared errors {np.mean(e ** 2) / np.mean(samp):.2f} (bootstrap 90% range {np.percentile(boot, 5):.2f}-{np.percentile(boot, 95):.2f}); "
      f"constant at the true mean {100 * R[:, 0].std():.2f}")

"""A much simpler estimator (2026-10-17): accuracy from three terms.

  accuracy = 1 - (8/9) * P(at most one positive input) - 2.54 * w23 - 1.54 * w12

- P(at most one positive of 10) = 11/1024 exactly. With one positive input every other input is negative and leaves no
  trace after one step, so the model can only guess among the nine positions other than the largest's (wrong 8 in 9).
- 2.54 and 1.54 are the densities at zero of the second-third and largest-second gaps for 10 standard normals (exact,
  model-free): the fraction of inputs within a gap w is about density * w.
- w23, w12: the model's average near-tie band widths. At one reference size (second = third = 0.9 with the largest 0.5
  above; largest = second = 0.9 with the third 0.4 below; the other seven at -0.46, the earlier estimator's settings,
  not re-chosen), take each ordered placement's tie margin M (right slot minus the competitor), the median slope of the
  margin per unit gap, and average max(0, -M) / slope over the 720 placements.
Ignores: overlap of the two bands, how band widths change with size, small-second and other mistakes.
Scored on all 22 trained models (all seen before; this formula was not tuned on them)."""
import glob
import sys

import numpy as np

sys.path.insert(0, "/data/output/research/algzoo/20261016_run3/scripts")
from band_origin import T, scores  # noqa: E402

from algzoo.core import example, load, zoo  # noqa: E402
from algzoo.prefix import cost_in_runs, tree_scores  # noqa: E402

n = len(T); ar = np.arange(n); B = -0.46; m = 0.9; h = 0.005
P1 = 11 / 1024


def widths(p):
    def X(m1, m2, m3):
        Z = np.full((n, 10), B); Z[ar, T[:, 0]] = m1; Z[ar, T[:, 1]] = m2; Z[ar, T[:, 2]] = m3
        return Z
    runs = [X(m + 0.5, m, m), X(m + 0.5, m, m - h), X(m, m, m - 0.4), X(m + h, m, m - 0.4)]
    S = [scores(p, Z) for Z in runs]
    M23 = S[0][ar, T[:, 1]] - S[0][ar, T[:, 2]]; s23 = np.median((S[1][ar, T[:, 1]] - S[1][ar, T[:, 2]] - M23) / h)
    M12 = S[2][ar, T[:, 1]] - S[2][ar, T[:, 0]]; s12 = np.median((S[3][ar, T[:, 1]] - S[3][ar, T[:, 0]] - M12) / h)
    _, steps, distinct = tree_scores(p, np.concatenate(runs))
    return np.mean(np.maximum(0, -M23)) / s23, np.mean(np.maximum(0, -M12)) / s12, cost_in_runs(p, steps, distinct)


truth6 = {"example": 0.9532, "seed0": 0.9101, "seed1": 0.8804, "seed2": 0.9485, "seed3": 0.9320, "seed4": 0.8898}
models = [("example", example())] + [(f"seed{s}", zoo(16, 10, s)) for s in range(5)]
new = {int(s): a for s, a in np.load("/data/output/research/algzoo/20261016_run3/reports/new_truth.npy")}
models += [(str(s), load(f"/data/output/research/algzoo/20261016_run3/models/2nd_argmax_16_10_1073741824_{s}.pth")) for s in sorted(new)]
rows = []
for name, p in models:
    a = truth6.get(name, new.get(int(name)) if name.isdigit() else None)
    w23, w12, cost = widths(p)
    terms = ((8 / 9) * P1, 2.54 * w23, 1.54 * w12)
    est = 1 - sum(terms)
    rows.append((a, est, cost))
    print(f"{name}: true {a:.4f}; estimate {est:.4f} ({100 * (est - a):+.2f} points); terms: one positive {100 * terms[0]:.2f}, "
          f"second-third {100 * terms[1]:.2f} (width {w23:.4f}), largest-second {100 * terms[2]:.2f} (width {w12:.4f}); cost {cost:.0f} runs' worth", flush=True)
R = np.array(rows); e = R[:, 1] - R[:, 0]
for lab, k in [("six original", slice(0, 6)), ("16 new", slice(6, None)), ("all 22", slice(None))]:
    samp = np.mean(R[k, 0] * (1 - R[k, 0]) / R[k, 2])
    print(f"{lab}: RMSE {100 * np.sqrt(np.mean(e[k] ** 2)):.2f} points, bias {100 * e[k].mean():+.2f}; correlation with truth {np.corrcoef(R[k, 0], R[k, 1])[0, 1]:.2f}; "
          f"sampling at the same cost {100 * np.sqrt(samp):.2f}; a constant at the true mean {100 * R[k, 0].std():.2f}")

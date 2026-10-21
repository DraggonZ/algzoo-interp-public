"""Surprise accounting for the simple estimator (2026-10-17). Judgement calls are stated, not hidden.

Total surprise = bits paid for the estimate's checks on the weights + surprise of the true accuracy given the estimate.
- Checks on the weights: a fixed part for the mechanism the formula relies on (B0, assumed 50 bits: signs and rough sizes
  of the input weights, the clock being hidden), plus each tie margin the formula looks at. A margin's sign matters
  (1 bit) and, when negative, its size to about 25% (about 4 more bits): about 3 bits per margin on average (assumption).
- Residual: the true accuracy (known to 0.003 points) given the estimate with its typical error sigma as uncertainty:
  log2(sigma * sqrt(2 pi) / 0.003 points) + z^2 / (2 ln 2) bits, averaged over models.
- Budget for comparison: the bits of optimization behind the model are at most the bits the weights take up. 432 weights
  at 32 bits are 13,824 bits; at the precision that matters (about 0.5% relative noise already costs points, so about
  8 bits each, fewer for weights near zero) a few thousand.
The formula's widths are averaged over fixed, evenly spaced subsets of the 720 placements (K = 720, 240, 90, 30, 10),
to trace how error grows as fewer margins are paid for. Scored on all 22 trained models (development, not pre-registered)."""
import sys

import numpy as np

sys.path.insert(0, "/data/output/research/algzoo/20261016_run3/scripts")
from simple_estimator import B, P1, T, ar, h, m, models, new, n, scores, truth6  # noqa: E402

B0, BITS_PER_MARGIN, RES = 50, 3, 0.003e-2
per = []
for name, p in models:
    def X(m1, m2, m3):
        Z = np.full((n, 10), B); Z[ar, T[:, 0]] = m1; Z[ar, T[:, 1]] = m2; Z[ar, T[:, 2]] = m3
        return Z
    S = [scores(p, Z) for Z in (X(m + 0.5, m, m), X(m + 0.5, m, m - h), X(m, m, m - 0.4), X(m + h, m, m - 0.4))]
    M23 = S[0][ar, T[:, 1]] - S[0][ar, T[:, 2]]; s23 = np.median((S[1][ar, T[:, 1]] - S[1][ar, T[:, 2]] - M23) / h)
    M12 = S[2][ar, T[:, 1]] - S[2][ar, T[:, 0]]; s12 = np.median((S[3][ar, T[:, 1]] - S[3][ar, T[:, 0]] - M12) / h)
    a = truth6.get(name, new.get(int(name)) if name.isdigit() else None)
    per.append((a, np.maximum(0, -M23) / s23, np.maximum(0, -M12) / s12))
print(f"{'placements used':>16} {'margins paid':>13} {'RMSE (points)':>14} {'check bits':>11} {'residual bits':>14} {'total bits':>11}")
for K in (720, 240, 90, 30, 10):
    idx = np.arange(0, 720, 720 // K)
    err = np.array([1 - (8 / 9) * P1 - 2.54 * b23[idx].mean() - 1.54 * b12[idx].mean() - a for a, b23, b12 in per])
    sig = np.sqrt(np.mean(err ** 2))
    resid = np.mean(np.log2(sig * np.sqrt(2 * np.pi) / RES) + (err / sig) ** 2 / (2 * np.log(2)))
    checks = B0 + BITS_PER_MARGIN * 2 * K
    print(f"{K:>16} {2 * K:>13} {100 * sig:>14.2f} {checks:>11.0f} {resid:>14.1f} {checks + resid:>11.0f}")

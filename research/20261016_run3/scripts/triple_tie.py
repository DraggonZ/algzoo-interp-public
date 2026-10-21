"""Do margins at a triple tie (m1 = m2 = m3, one run per unordered position set) predict the near-tie bands at real gaps?

For each ordered placement and m1 - m2 = g12 in {0.2, 0.5, 1.0} (m2 = m3 = 0.9, others -0.46): the second-third tie margin
M0(g12) against the triple-tie margin score[p2] - score[p3] at m1 = m2 = m3 = 0.9. Then the mean second-third band
max(0, -M) / slope from each, against the measured band (band_origin.py's scan).
"""
import itertools
import sys

import numpy as np

sys.path.insert(0, "/data/output/research/algzoo/20261016_run3/scripts")
from band_origin import G23, T, b, scores  # noqa: E402

from algzoo.core import example, zoo  # noqa: E402

n = len(T); ar = np.arange(n)
for name in sys.argv[1:] or ["example", 2, 3, 0, 4, 1]:
    p = example() if name == "example" else zoo(16, 10, int(name))
    m2 = 0.9
    X = np.full((n, 10), b); X[ar, T[:, 0]] = m2; X[ar, T[:, 1]] = m2; X[ar, T[:, 2]] = m2
    S3 = scores(p, X); Mt = S3[ar, T[:, 1]] - S3[ar, T[:, 2]]          # same run for all orderings of a set
    for g12 in (0.2, 0.5, 1.0):
        m1 = m2 + g12
        Xg = np.full((n, len(G23), 10), b)
        Xg[ar, :, T[:, 0]] = m1; Xg[ar, :, T[:, 1]] = m2
        Xg[ar[:, None], np.arange(len(G23))[None, :], T[:, 2][:, None]] = m2 - G23[None, :]
        S = scores(p, Xg.reshape(-1, 10)).reshape(n, len(G23), 10)
        band = (S.argmax(-1) != T[:, 1:2])[:, 1:].sum(1) * 0.001
        M0 = S[ar, 0, T[:, 1]] - S[ar, 0, T[:, 2]]
        slope = np.median((S[ar, 5, T[:, 1]] - S[ar, 5, T[:, 2]] - M0) / 0.005)
        pt = np.maximum(0, -Mt) / slope; p0 = np.maximum(0, -M0) / slope
        print(f"{name} g12={g12}: corr(tie margin, triple-tie margin) {np.corrcoef(M0, Mt)[0, 1]:.2f}; same sign {np.mean(np.sign(M0) == np.sign(Mt)):.2f}; "
              f"mean band measured {band.mean():.4f}, from tie margin {p0.mean():.4f}, from triple tie {pt.mean():.4f} "
              f"(corr with measured {np.corrcoef(pt, band)[0, 1]:.2f})", flush=True)

"""Shape of the reduction's mistake regions, from the dense quadrature grids (reduce3_quad.py output).

For each triple and value of the second (m2): is "right" the AND of a 1-2 condition (depends on m1 only) and a 2-3 condition
(depends on m3 only)? Test: right(u, v) == right(u, v_mid) & right(u_mid, v). Report the weight of mismatches, and how the
mistake weight splits between a 1-2 band (wrong at u small), a 2-3 band (v near 1) and whole-slice failures.
"""
import sys

import numpy as np

sys.path.insert(0, "/data/output/research/algzoo/20261016_run3/scripts")
from reduce3_quad import WU, WV  # noqa: E402

for name in ["example", "0", "1", "2", "3", "4"]:
    d = np.load(f"reports/reduce3_quad_{name}.npz")
    r, wt = d["right"], d["wt"]          # [720, N2, NU, NV]
    W = 720 * wt[None, :, None, None] * WU[None, None, :, None] * WV[None, None, None, :] / 720
    iu, iv = int(sys.argv[1]) if len(sys.argv) > 1 else len(WU) - 1, int(sys.argv[2]) if len(sys.argv) > 2 else 0   # references far from each tie
    sep = r[:, :, :, iv:iv + 1] & r[:, :, iu:iu + 1, :]
    mis = (sep != r)
    wrong = ~r
    whole = wrong.all((2, 3))[:, :, None, None] & wrong
    # 1-2 band: wrong where u small but right at u_mid (same v); 2-3 band: wrong at v and right at v_mid
    b12 = wrong & r[:, :, iu:iu + 1, :]
    b23 = wrong & r[:, :, :, iv:iv + 1]
    tot = (W * wrong).sum()
    print(f"{name}: wrong {100 * tot:.2f} pts; separable model off by {100 * (W * mis).sum():.3f} pts "
          f"(of which predicts wrong but right {100 * (W * (mis & r)).sum():.3f}); whole-slice {100 * (W * whole).sum():.2f}, "
          f"1-2 band only {100 * (W * (b12 & ~b23)).sum():.2f}, 2-3 band only {100 * (W * (b23 & ~b12)).sum():.2f}, "
          f"both {100 * (W * (b12 & b23)).sum():.2f}, other {100 * (W * (wrong & ~b12 & ~b23 & ~whole)).sum():.2f}")

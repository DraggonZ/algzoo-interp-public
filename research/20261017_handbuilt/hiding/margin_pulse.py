"""On near-tie overtaking lists (margin_route.sel): how big is the detector's pulse at the overtaking, D(q) = h_4(q),
against the true jump 10.16 (x_q - x_r)? The detector compares x_q with the faded store, so its threshold
theta = x_q - D(q) / 10.16 lies below x_r. Also: the share of lists where the overtaking leaves some of the four on
(partial wipe), and the late gate's value right after (h_9(q+1)) against D(q). Medians."""
import numpy as np
from common import WI, WH, WO, C, lists, run
from margin_route import sel
for rr, qq in ((1, 4), (2, 5), (3, 6), (4, 7)):
    Xs = sel(rr, qq); L0, P0, H0 = run(Xs); J = Xs[:, qq] - Xs[:, rr]; D = H0[:, qq, 4]
    theta = Xs[:, qq] - D / WI[4]; part = (H0[:, qq][:, C] > 0).any(1)
    M = L0[:, rr] - L0[:, qq]
    print(f"record at {rr}, overtaken at {qq}: D(q) / (10.16 jump) median {np.median(D / (WI[4] * J)):.1f}; detector threshold / x_r {np.median(theta / Xs[:, rr]):.3f}; "
          f"partial wipe at q on {part.mean():.2f} of lists (margin median {np.median(M[part]):.1f}, right {np.mean(L0[part].argmax(1) == rr):.3f}) vs full wipe "
          f"(margin {np.median(M[~part]):.1f}, right {np.mean(L0[~part].argmax(1) == rr):.3f}); late gate at q+1 / D(q) median {np.median(H0[:, qq + 1, 9] / np.maximum(D, 1e-9)):.3f}")

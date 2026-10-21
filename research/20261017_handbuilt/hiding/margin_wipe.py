"""Near-tie overtakings (margin_route.sel), split by whether the overtaking silences all four hiding neurons at q (full
wipe) or leaves some on (partial wipe). Exact derivatives within each run's pattern (backpropagation): the margin's
change per unit of the old record x_r and of the new largest x_q (medians), and the late gate's share of the margin's
level dependence at q+1. Then the late gate's value right after the overtaking, h_9(q+1), fitted on (x_r, x_q) and on
(x_r, jump) by least squares (no intercept), and its parts: 0.15 x detector pulse, memory, the four, input."""
import numpy as np
from common import WI, WH, WO, C, GROUPS, lists, run
from margin_route import sel
from margin_carry import fwd, adj
for rr, qq in ((1, 4), (2, 5), (3, 6), (4, 7)):
    Xs = sel(rr, qq); L0, P0, H0 = run(Xs); B = len(Xs); full = (H0[:, qq][:, C] == 0).all(1)
    a = adj(P0, np.broadcast_to(WO[rr] - WO[qq], (B, 16))); g = (a * (P0 > 0) * WI).sum(2)   # dM/dx_t
    dx = np.zeros((B, 10)); dx[:, [rr, qq]] = 1; dH = fwd(P0, dx); late = a[:, qq + 1, 9] * dH[:, qq + 1, 9]
    out = []
    for lab, m in (("full wipe", full), ("partial wipe", ~full)):
        out.append(f"{lab} ({m.mean():.2f}): dM/dx_r {np.median(g[m, rr]):+7.1f}, dM/dx_q {np.median(g[m, qq]):+7.1f}, level {np.median(g[m, rr] + g[m, qq]):+6.1f} (late gate's share {np.median(late[m]):+5.1f}), margin {np.median(L0[m, rr] - L0[m, qq]):5.1f}")
    print(f"record at {rr}, overtaken at {qq}: " + " | ".join(out))
    h9 = H0[:, qq + 1, 9]; J = Xs[:, qq] - Xs[:, rr]
    for lab, A in (("x_r, x_q", np.c_[Xs[:, rr], Xs[:, qq]]), ("x_r, jump", np.c_[Xs[:, rr], J])):
        co = np.linalg.lstsq(A, h9, rcond=None)[0]; e = h9 - A @ co
        print(f"   late gate at q+1 = {co[0]:+.2f} x ({lab.split(',')[0]}) {co[1]:+.2f} x ({lab.split(', ')[1]})  (variance matched {1 - e.var() / h9.var():.2f}); median value {np.median(h9):.2f}")
    parts = {g_: H0[:, qq][:, idx] @ WH[9, idx] for g_, idx in GROUPS.items()}
    print("   its input at q+1, medians: " + ", ".join(f"{g_} {np.median(v):+.2f}" for g_, v in parts.items()) + f", input {np.median(WI[9] * Xs[:, qq + 1]):+.2f}")

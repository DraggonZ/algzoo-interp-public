"""Continuous steering along the shift register. Replace the final rotating state h_rot by M^a h_rot, M the rotating
block's weights (fractional powers through its eigenvalues; the readout then sees votes a steps older). Predictions
(written first): a = 1 names second-1 on most lists (like an extra step, but without the ReLU and the other neurons);
a = 0.5 splits answers between the second and second-1; a = -1 (rotating back) names second+1."""
import numpy as np
from harness import *
from modes import ROT, ev, R, Linv, X, S, h, a0
def power(a):
    return (R @ np.diag(ev.astype(complex) ** a) @ Linv).real
for a in (0.25, 0.5, 0.75, 1.0, -0.5, -1.0):
    M = torch.tensor(power(a), device=DEV, dtype=dt); h2 = h.clone(); h2[:, ROT] = h[:, ROT] @ M.T; ans = (h2 @ ARC[2].T).argmax(1)
    f = lambda d: 100 * (ans[(S + d >= 0) & (S + d <= 9)] == (S + d)[(S + d >= 0) & (S + d <= 9)]).double().mean().item()
    print(f"a {a:+.2f}: names second-1 {f(-1):5.1f}% | the second {f(0):5.1f}% | second+1 {f(1):5.1f}%", flush=True)

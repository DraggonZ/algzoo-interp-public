"""r2b diagnostic: is the previous-jump term a correction for hiding neurons that switched off at a record?
One-step least squares with ARC's previous hiding states (blind feedback), on-steps only, as r1b_feedback.py.
Variants of the feedback source h_k(t-1):
  clipped:          ARC's activity (the masked rule)
  unclipped@rec:    ARC's pre-activation (negative allowed) on steps right after a record, activity otherwise
  unclipped:        ARC's pre-activation always
each with and without the previous-jump feature. Prints one-step rms per neuron and the fitted jump coefficients."""
import sys
sys.path.insert(0, "rules")
import numpy as np
from common import HID, WI, lag, task_vars
import simlib
from r1b_masked import feats, MASK

X = simlib.lists(101, 50_000); _, P, H = simlib.run(X)
Fn = feats(X); V = task_vars(X)
rec_prev = lag(V["newmax"].astype(float)) > 0        # step t-1 was a record
Hp = np.concatenate([np.zeros((len(X), 1, 16)), H[:, :-1]], 1)[..., HID]
Pp = np.concatenate([np.zeros((len(X), 1, 16)), P[:, :-1]], 1)[..., HID]
srcs = {"clipped": Hp, "unclipped@rec": np.where(rec_prev[..., None], Pp, Hp), "unclipped": Pp}
for name, S in srcs.items():
    blind = S - lag(X)[..., None] * WI[HID]
    for jump in [True, False]:
        out = []
        for i, n in enumerate(HID):
            m = MASK[i].copy()
            if not jump: m[3] = False
            y = (P[:, :, n] - WI[n] * X).reshape(-1); on = (P[:, :, n] > 0).reshape(-1)
            A = np.concatenate([Fn[..., m], blind], -1).reshape(-1, m.sum() + 4)[on]
            c = np.linalg.lstsq(A, y[on], rcond=None)[0]
            r = np.sqrt(((A @ c - y[on]) ** 2).mean())
            jc = c[int(m[:3].sum())] if m[3] else float("nan")
            out.append(f"{r:.3f} (jump {jc:+.2f}; fb {np.array2string(c[-4:], precision=2)})")
        print(f"{name:14s} jump={'yes' if jump else 'no '}: " + " | ".join(out))

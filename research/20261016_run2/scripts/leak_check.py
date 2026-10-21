"""Find where M2,5 loses mass: measure every split cell before and after the cut (forward steps, order cuts, logit cuts)."""
import numpy as np

import exact_nd as E
from algzoo.core import load

orig_cut, orig_measure = E.cut, E.cone_measure
log = []


def cut_dbg(R, H, b, ops):
    p, m = orig_cut(R, H, b, ops)
    s = R @ (b / np.linalg.norm(b))
    if (s > E.TOL).any() and (s < -E.TOL).any():
        before = orig_measure(R, E.Ops())
        after = sum(orig_measure(c[0], E.Ops()) for c in (p, m) if c is not None)
        if abs(before - after) > 1e-9:
            log.append((before - after, len(R), int((np.abs(s) <= E.TOL).sum()), None if p is None else len(p[0]),
                        None if m is None else len(m[0]), float(np.sort(np.abs(s))[:3].max())))
    return p, m


E.cut = cut_dbg
acc, st = E.exact_accuracy(load("2nd_argmax_2_5_1073741824_bo5.pth"))
lost = sum(x[0] for x in log)
print(f"acc {acc:.8f}; splits losing > 1e-9: {len(log)}; total lost {lost:.3e}")
log.sort(reverse=True)
for x in log[:15]:
    print("lost %.2e rays %d zero-side rays %d plus %s minus %s smallest-|s| %.1e" % x)

"""Split the effect of adding the second number into the comparing neurons' view and the other ten's view."""
import numpy as np

from algzoo.core import example

w_hi, w_hh, w_oh = example()
C = [1, 2, 4, 6, 7, 8]
Dn = [i for i in range(16) if i not in C]
Wc = w_oh - w_oh.mean(0)
r = 0.7


def run(x):
    h = np.zeros(16)
    for t in range(10):
        h = np.maximum(0, w_hh @ h + x[t] * w_hi)
    return h


for lab, cond in [("max first", lambda a, b: a < b), ("2nd first", lambda a, b: a > b)]:
    stats = []
    for p1 in range(10):
        for p2 in range(10):
            if p1 == p2 or not cond(p1, p2):
                continue
            x1 = np.zeros(10); x1[p1] = 1.0
            x2 = x1.copy(); x2[p2] = r
            hb, hm = run(x2), run(x1)
            dc = Wc[:, C] @ (hb - hm)[C]; dd = Wc[:, Dn] @ (hb - hm)[Dn]
            fc = Wc[:, C] @ hb[C]; fd = Wc[:, Dn] @ hb[Dn]
            stats.append((p1, p2, np.linalg.norm(dc), np.linalg.norm(dd), np.corrcoef(dc, dd)[0, 1],
                          dc.argmax() == p2, dd.argmax() == p2, (dc + dd).argmax() == p2,
                          np.linalg.norm(fc), np.linalg.norm(fd), np.linalg.norm(fc + fd), fd.argmax() == p2, (-fc).argmax() == p2))
    S = np.array([s[2:] for s in stats], dtype=float)
    print(f"{lab} ({len(stats)} pairs):")
    print(f"  change from adding the second: size in comparing view {np.median(S[:,0]):.1f}, in other view {np.median(S[:,1]):.1f}, median corr {np.median(S[:,2]):+.2f}")
    print(f"  points at second's position: comparing view {int(S[:,3].sum())}, other view {int(S[:,4].sum())}, both {int(S[:,5].sum())}")
    print(f"  final state: size comparing view {np.median(S[:,6]):.1f}, other view {np.median(S[:,7]):.1f}, sum {np.median(S[:,8]):.1f}")
    print(f"  final other view alone points at second: {int(S[:,9].sum())}; final comparing view's minimum at second: {int(S[:,10].sum())}")

"""What sets the store's weight on the previous largest P? Exact per-list weights (store_jacobian.py). By the age b of
the largest L, split by the gap g = (steps between P's arrival and L's arrival) and by P/L."""
import numpy as np
exec(open("store_jacobian.py").read().split('print("age |')[0])
for a in (2, 4, 6, 8):
    rows = [(t, np.where((R[:, t] == t - a) & (RP[:, t] >= 0) & (Xn[ar, np.maximum(R[:, t], 0)] > 0.3))[0]) for t in range(a, 10)]
    cP = np.concatenate([JS[idx, t, RP[idx, t]] for t, idx in rows])
    g = np.concatenate([(t - a) - RP[idx, t] for t, idx in rows])
    ratio = np.concatenate([Xn[idx, RP[idx, t]] / Xn[idx, t - a] for t, idx in rows])
    print(f"age {a}: weight on P by gap between P and L (median, n): " + "  ".join(f"g={k}: {np.median(cP[g == k]):+.2f} ({(g == k).sum()})" for k in range(1, 7) if (g == k).sum() > 50))
    print(f"        by P/L: " + "  ".join(f"[{lo:.1f},{hi:.1f}) {np.median(cP[(ratio >= lo) & (ratio < hi)]):+.2f}" for lo, hi in [(0, .5), (.5, .7), (.7, .8), (.8, .9), (.9, 1.0)] if ((ratio >= lo) & (ratio < hi)).sum() > 50))

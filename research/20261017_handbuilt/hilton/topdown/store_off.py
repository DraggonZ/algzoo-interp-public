"""When does the store switch off late after a record? Per age, share of lists with the store at 0, and how that
depends on the previous largest relative to the largest (P/L), and on whether a number since was kept."""
import numpy as np
exec(open("store_jacobian.py").read().split('print("age |')[0])
for a in range(4, 9):
    rows = [(t, np.where((R[:, t] == t - a) & (R[:, t] >= 0) & (Xn[ar, np.maximum(R[:, t], 0)] > 0.3))[0]) for t in range(a, 10)]
    off = np.concatenate([S[idx, t] <= 0 for t, idx in rows])
    L_ = np.concatenate([Xn[idx, t - a] for t, idx in rows])
    P_ = np.concatenate([np.where(RP[idx, t] >= 0, Xn[idx, np.maximum(RP[idx, t], 0)], 0) for t, idx in rows])
    ratio = P_ / L_
    bins = [-9, 0, 0.5, 0.7, 0.8, 0.9, 1.0]
    print(f"age {a}: store off on {100*off.mean():.1f}% | by P/L: " + "  ".join(f"[{lo:.1f},{hi:.1f}) {100*off[(ratio>=lo)&(ratio<hi)].mean():.0f}%" for lo, hi in zip(bins[:-1], bins[1:])))
    # numbers since: max since / L
    ms = np.concatenate([Xn[idx][:, t - a + 1:t].max(1) if a > 1 else np.zeros(len(idx)) for t, idx in rows]) / L_
    b2 = [-9, 0, 0.5, 0.7, 0.8, 0.9, 1.0]
    print(f"        by the largest number since / L: " + "  ".join(f"[{lo:.1f},{hi:.1f}) {100*off[(ms>=lo)&(ms<hi)].mean():.0f}%" for lo, hi in zip(b2[:-1], b2[1:])))

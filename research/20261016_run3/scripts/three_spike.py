"""How is a new input compared with the current second? Three spikes: largest 1, second m, newcomer x (rest zero).

At the step the newcomer arrives, margin = score at the newest slot minus score at the second's slot. If votes are
proportional to value, margin = c * (x - m) and crosses zero at x = m. Reports the crossing point and the slope near it.
"""
import numpy as np

from algzoo.core import example

w_hi, w_hh, w_oh = (np.asarray(p, np.float64) for p in example())


def scores_at(x, t_stop):
    h = np.zeros(16)
    for t in range(t_stop + 1):
        h = np.maximum(0, w_hh @ h + x[t] * w_hi)
    s = w_oh @ h
    return s - s.mean()


rows = []
for pM, pm, px in [(0, 2, 5), (0, 5, 7), (3, 1, 6), (1, 4, 9), (6, 2, 8), (2, 6, 7), (0, 1, 9), (4, 7, 8)]:
    for m in (0.4, 0.7):
        xs = np.linspace(m - 0.2, m + 0.2, 81)
        marg = []
        for xv in xs:
            x = np.zeros(10); x[pM] = 1.0; x[pm] = m; x[px] = xv
            s = scores_at(x, px)
            marg.append(s[9] - s[pm + 9 - px])
        marg = np.array(marg)
        k = np.where(np.diff(np.sign(marg)) != 0)[0]
        cross = xs[k[0]] if len(k) else np.nan
        slope = np.polyfit(xs[30:51], marg[30:51], 1)[0]
        rows.append((pM, pm, px, m, cross - m, slope))
        print(f"largest at {pM}, second {m} at {pm}, newcomer at {px}: margin crosses zero at x - m = {cross - m:+.3f}; slope {slope:.0f} per unit")

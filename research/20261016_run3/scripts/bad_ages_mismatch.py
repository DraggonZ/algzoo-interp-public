"""Review 5's lead: does event 1's mismatch (actual change versus the simple linear vote, on the right-slot margin) at
each age of the largest predict the ages at which a copy loses the vote?

For each copy and age g = 1..8 of the largest when a second (0.7) arrives: actual change in centered scores versus the
linear vote r * (output weights of on, negatively weighted neurons) @ (their input weights); mismatch = |actual margin
change - linear margin change| on the newest slot minus best other. Losses by age as in bad_ages.py.
"""
from collections import Counter

import numpy as np
from scipy.stats import spearmanr

from algzoo.bench import load_bench

names, acc, models = load_bench(("mult",))
cells = []
for nm, prm in zip(names, models):
    w_hi, w_hh, w_oh = (np.asarray(q, np.float64) for q in prm)
    Wc = w_oh - w_oh.mean(0)
    h = np.maximum(0, w_hi); pres = [w_hi]
    for k in range(1, 10):
        pre = w_hh @ h; pres.append(pre); h = np.maximum(0, pre)
    mism = {}
    for g in range(1, 9):
        pre = pres[g]
        actual = Wc @ (np.maximum(0, pre + 0.7 * w_hi) - np.maximum(0, pre))
        on = [i for i in range(16) if pre[i] > 0 and w_hi[i] < 0]
        lin = 0.7 * (Wc[:, on] @ w_hi[on])
        ma = actual[9] - np.delete(actual, 9).max(); ml = lin[9] - np.delete(lin, 9).max()
        mism[g] = abs(ma - ml)
    loss = Counter()
    for r in (0.3, 0.7, 0.95):
        for p1 in range(10):
            for p2 in range(p1 + 1, 10):
                x = np.zeros(10); x[p1] = 1.0; x[p2] = r
                hh = np.zeros(16); S = []
                for t in range(10):
                    hh = np.maximum(0, w_hh @ hh + x[t] * w_hi); s = w_oh @ hh; S.append(s - s.mean())
                for t in range(p2 + 1, 10):
                    if S[t - 1].argmax() == p2 + 9 - (t - 1) and S[t].argmax() != p2 + 9 - t:
                        loss[p2 - p1] += 1      # credit the loss to the largest's age when the second arrived
                        break
    for g in range(1, 9):
        cells.append((str(nm), g, mism[g], loss[g]))
C = np.array([(c[2], c[3]) for c in cells])
print(f"(copy, age) cells: {len(C)}; Spearman between event-1 mismatch at that age and losses of seconds that arrived at that age: {spearmanr(C[:, 0], C[:, 1])[0]:+.2f}")
per = []
for nm in set(c[0] for c in cells):
    cc = np.array([(c[2], c[3]) for c in cells if c[0] == nm])
    if cc[:, 1].sum() >= 5:
        per.append(spearmanr(cc[:, 0], cc[:, 1])[0])
print(f"within copies with at least 5 losses ({len(per)}): median Spearman {np.nanmedian(per):+.2f}")

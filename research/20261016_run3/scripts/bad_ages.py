"""Do degraded copies drop the vote at particular ages of the largest's clock, and do the weights predict which?

Behaviour: in two-spike runs (90 pairs x ratios 0.3, 0.7, 0.95), count vote losses by the largest's age at the step of loss.
Weights: for each age a of a lone largest, take the step matrix with the copy's own on/off pattern at that age (the
pattern the slide uses when the largest is a steps old) and measure the slide relation (median cosine of output row j vs
row j-1 times that matrix). Compare across copies and ages.
"""
from collections import Counter

import numpy as np
from scipy.stats import spearmanr

from algzoo.bench import load_bench

names, acc, models = load_bench(("mult",))
cells = []
conc = []
for nm, a_, prm in zip(names, acc, models):
    w_hi, w_hh, w_oh = (np.asarray(q, np.float64) for q in prm)
    Wc = w_oh - w_oh.mean(0)
    loss = Counter(); tot = Counter()
    for r in (0.3, 0.7, 0.95):
        for p1 in range(10):
            for p2 in range(10):
                if p1 == p2:
                    continue
                x = np.zeros(10); x[p1] = 1.0; x[p2] = r
                h = np.zeros(16); S = []
                for t in range(10):
                    h = np.maximum(0, w_hh @ h + x[t] * w_hi); s = w_oh @ h; S.append(s - s.mean())
                t0 = max(p1, p2)
                for t in range(t0 + 1, 10):
                    tot[t - p1] += 1
                    if S[t - 1].argmax() == p2 + 9 - (t - 1) and S[t].argmax() != p2 + 9 - t:
                        loss[t - p1] += 1
                        break
    h = np.maximum(0, w_hi); pres = [w_hi]
    for k in range(1, 10):
        pre = w_hh @ h; pres.append(pre); h = np.maximum(0, pre)
    for age in range(1, 10):
        M = (pres[age] > 0).astype(float)[:, None] * w_hh
        cs = np.median([np.dot(Wc[j], Wc[j - 1] @ M) / (np.linalg.norm(Wc[j]) * np.linalg.norm(Wc[j - 1] @ M) + 1e-12) for j in range(1, 10)])
        cells.append((str(nm), age, loss[age] / max(tot[age], 1), cs))
    L = np.array([loss[a] for a in range(1, 10)])
    if L.sum() >= 10:
        conc.append((str(nm), L.sum(), np.sort(L)[-2:].sum() / L.sum(), int(np.argmax(L)) + 1))
C = np.array([(c[2], c[3]) for c in cells])
print(f"copies with at least 10 losses: {len(conc)}; share of losses at the copy's two worst ages: median {np.median([c[2] for c in conc]):.2f}")
print("worst age of the largest per copy:", Counter(c[3] for c in conc))
print(f"across (copy, age) cells: Spearman between loss rate and the slide relation at that age {spearmanr(C[:, 0], C[:, 1])[0]:+.2f}")

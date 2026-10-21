"""Review 5 check C: how many linear regions (on/off patterns over all 10 steps) carry ARC's model's mistakes on full inputs?

Diagnostic only (sampled). If the error mass is spread over far more regions than a budget of runs, any estimator that
works region by region cannot match sampling at that budget.
"""
import sys
from collections import Counter

import numpy as np

from algzoo.core import example, zoo

name = sys.argv[1] if len(sys.argv) > 1 else "example"
params = example() if name == "example" else zoo(16, 10, int(name))
w_hi, w_hh, w_oh = (np.asarray(p, np.float64) for p in params)
rng = np.random.default_rng(11)
cnt_wrong = Counter(); cnt_all = Counter()
n_tot = 0; n_wrong = 0
for chunk in range(8):
    N = 1 << 18
    X = rng.standard_normal((N, 10))
    hh = np.zeros((N, 16)); bits = []
    for t in range(10):
        hh = np.maximum(0, hh @ w_hh.T + X[:, t, None] * w_hi)
        bits.append(hh > 0)
    B = np.packbits(np.concatenate(bits, 1), axis=1)
    pred = (hh @ w_oh.T).argmax(1)
    p2 = np.argsort(X, 1)[:, -2]
    wrong = pred != p2
    keys = [b.tobytes() for b in B]
    for k, w in zip(keys, wrong):
        cnt_all[k] += 1
        if w:
            cnt_wrong[k] += 1
    n_tot += N; n_wrong += wrong.sum()
print(f"model {name}: {n_tot} inputs, accuracy {1 - n_wrong / n_tot:.4f}")
print(f"distinct regions seen: all inputs {len(cnt_all)}, wrong inputs {len(cnt_wrong)}")
c = np.sort(np.array(list(cnt_wrong.values())))[::-1]
cs = np.cumsum(c) / c.sum()
for f in [0.25, 0.5, 0.75, 0.9]:
    print(f"  regions needed to cover {f:.0%} of sampled mistakes: {np.searchsorted(cs, f) + 1}")
print(f"  regions with exactly one sampled mistake: {np.sum(c == 1)} ({np.sum(c == 1) / c.sum():.0%} of mistakes)")
mixed = sum(1 for k in cnt_wrong if cnt_all[k] > cnt_wrong[k])
print(f"  regions holding mistakes that also hold right answers (decision boundary inside): {mixed} of {len(cnt_wrong)}")

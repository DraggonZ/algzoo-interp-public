"""Stronger control for the sufficiency test (scripts/silencing_random_lists.py, part 1): silence every one of the 495
sets of four among the twelve other neurons, instead of 10 random sets, on 50,000 random lists whose largest is positive
and arrives before the last step; silencing at a random later step. Reports how often the largest's position wins."""
import itertools
import numpy as np
from algzoo.core import example
wi, wh, wo = (np.asarray(w, float) for w in example()); wi = wi.reshape(-1); LOW = [1, 6, 7, 8]; OTHER = [k for k in range(16) if k not in LOW]
rng = np.random.default_rng(0)
X = rng.standard_normal((80_000, 10)); p1 = X.argmax(1); s = (X.max(1) > 0) & (p1 <= 8); X, p1 = X[s][:50_000], p1[s][:50_000]
t_set = np.array([rng.integers(a + 1, 10) for a in p1])


def win(G):
    h = np.zeros((len(X), 16))
    for t in range(10):
        h = np.maximum(0, h @ wh.T + X[:, t:t + 1] * wi); m = t_set == t; h[np.ix_(m, G)] = 0
    return np.mean((h @ wo.T).argmax(1) == p1)


base = win([]); four = win(LOW)
res = np.array([win(list(G)) for G in itertools.combinations(OTHER, 4)])
print(f"{len(X)} lists. Largest wins: no intervention {base:.3f}; silencing 1, 6, 7, 8 {four:.3f}")
print(f"other sets of four (all {len(res)}): median {np.median(res):.3f}, 95th percentile {np.percentile(res, 95):.3f}, max {res.max():.3f}")
best = list(itertools.combinations(OTHER, 4))[int(res.argmax())]; print("best other set:", best)

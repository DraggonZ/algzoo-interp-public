"""Shape of the vote on full random inputs: centered score at an input's own slot, by its rank and value, by age."""
import numpy as np

from algzoo.core import example

w_hi, w_hh, w_oh = (np.asarray(p, np.float64) for p in example())
Wc = w_oh - w_oh.mean(0)
rng = np.random.default_rng(8)
X = rng.standard_normal((400000, 10))
hh = np.zeros((len(X), 16))
for t in range(10):
    hh = np.maximum(0, hh @ w_hh.T + X[:, t, None] * w_hi)
S = hh @ Wc.T
rank = np.argsort(np.argsort(-X, 1), 1)          # 0 = largest
print("mean centered score at the input's own slot, by its rank (rows) and age (columns 0..9):")
for rk in range(5):
    row = []
    for a in range(10):
        k = 9 - a
        m = rank[:, k] == rk
        row.append(S[m, k].mean())
    print(f"  rank {rk + 1}: " + " ".join(f"{v:7.1f}" for v in row))
print("\nscore of the second-largest's slot minus the best other slot (margin), by the second's age, median and 10th pct:")
p2 = np.argsort(X, 1)[:, -2]
Sm = S.copy(); Sm[np.arange(len(X)), p2] = -np.inf
marg = S[np.arange(len(X)), p2] - Sm.max(1)
for a in range(10):
    m = p2 == 9 - a
    print(f"  age {a}: median {np.median(marg[m]):6.1f}, 10th pct {np.percentile(marg[m], 10):6.1f}, accuracy {np.mean(marg[m] > 0):.3f}")
print("\nby value of the second-largest (all ages): median margin")
v2 = X[np.arange(len(X)), p2]
for lo, hi in [(-3, 0), (0, 0.5), (0.5, 1), (1, 1.5), (1.5, 2), (2, 4)]:
    m = (v2 >= lo) & (v2 < hi)
    print(f"  second in [{lo},{hi}): median margin {np.median(marg[m]):6.1f}, accuracy {np.mean(marg[m] > 0):.3f}, share {m.mean():.3f}")

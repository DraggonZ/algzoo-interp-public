"""ARC: the swap test on full random lists. Base list x: random normal, second-largest at a, largest at p1 > a.
Donor: x with the values at a and a2 swapped (a2 < p1, a2 != a), so the donor's second-largest sits at a2 and the
two lists agree from step p1 on. Copy a neuron group from the donor into the base run at step p1 + k and see whether
the answer moves to the donor network's answer."""
import numpy as np

from algzoo.core import example

wi, wh, wo = example()
L = [1, 6, 7, 8]; R = [k for k in range(16) if k not in L]
rng = np.random.default_rng(0)


def run(X, patch_t=None, G=None, HD=None):
    """X: (n, 10). Returns answers and all states (n, 10, 16). Patch group G at step patch_t from HD (n, 10, 16)."""
    h = np.zeros((len(X), 16)); hs = []
    for t in range(10):
        h = np.maximum(0, h @ wh.T + X[:, t:t + 1] * wi)
        if patch_t is not None:
            m = patch_t == t
            h[np.ix_(m, G)] = HD[m, t][:, G]
        hs.append(h)
    return (h @ wo.T).argmax(1), np.stack(hs, 1)


n = 200_000
X = rng.standard_normal((n, 10))
order = np.argsort(-X, 1); p1, a = order[:, 0], order[:, 1]
keep = (a < p1) & (p1 >= 2)
X, p1, a = X[keep], p1[keep], a[keep]
a2 = np.array([rng.choice([j for j in range(q) if j != b]) for q, b in zip(p1, a)])
D = X.copy(); i = np.arange(len(X)); D[i, a], D[i, a2] = X[i, a2], X[i, a]
base, _ = run(X); donor, HD = run(D)
ok = (base == a) & (donor == a2)          # the network is right on both lists
print(f"lists with second-largest before largest: {len(X)}; network right on both base and donor: {ok.mean():.3f}")
groups = {"twelve": R, "four (1, 6, 7, 8)": L, "all 16": list(range(16))}
rand12 = [sorted(rng.choice(16, 12, replace=False)) for _ in range(5)]
for k in (0, 1, 2, 4):
    m = ok & (p1 + k <= 9)
    line = []
    for lab, G in list(groups.items()) + [(f"random twelve {j}", g) for j, g in enumerate(rand12)]:
        ans, _ = run(X[m], (p1 + k)[m], G, HD[m])
        line.append(f"{lab}: {np.mean(ans == donor[m]):.2f}")
    print(f"k={k} (n={m.sum()}): answer moves to donor's when copying " + "; ".join(line))
# Break the k=0 twelve result down by how far below the second-largest the third-largest is.
m = ok
ans, _ = run(X[m], p1[m], R, HD[m])
third = -np.sort(-X[m], 1)[:, 2]; gap = (X[m][np.arange(m.sum()), a[m]] - third)
for lo, hi in ((0, .05), (.05, .2), (.2, .5), (.5, 9)):
    s = (gap >= lo) & (gap < hi)
    print(f"  second minus third in [{lo}, {hi}): n={s.sum()}, twelve moves answer {np.mean(ans[s] == donor[m][s]):.3f}")

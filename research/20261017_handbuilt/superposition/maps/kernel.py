"""Test 7, see README.md."""
import numpy as np, torch as th
from collections import defaultdict
from algzoo.core import example
P = [th.tensor(np.asarray(p, float)) for p in example()]; wi, wh, wo = P
rng = np.random.default_rng(12); L = []
while len(L) < 4000:
    x = rng.standard_normal(10); v = np.sort(x)[::-1]
    if v[1] > 0 and v[1] - v[2] < 0.05 * v[1]: L.append(x)
X = np.array(L); O = np.argsort(-X, 1)
x = th.tensor(X, requires_grad=True); h = th.zeros(len(X), 16, dtype=th.float64); H = []
for t in range(10): h = th.relu(h @ wh.T + x[:, t:t + 1] * wi); H.append(h)
s = h @ wo.T; n = th.arange(len(X)); lead = s[n, th.tensor(O[:, 1])] - s[n, th.tensor(O[:, 2])]
g, = th.autograd.grad(lead.sum(), x); g = g.numpy(); H = th.stack(H, 1).detach().numpy()
rows = []
for i in range(len(X)):
    for p in O[i, 3:]:
        if p < 9 and H[i, p + 1, 5] > 0: rows.append((i, p - O[i, 1], p - O[i, 2], p - O[i, 0], g[i, p]))
R = np.array(rows, dtype=float); half = R[:, 0] < 2000; print(f"kept smaller numbers: {len(R)} ({half.sum()} fit, {(~half).sum()} scored)")
def fit_score(keys):
    tab = defaultdict(list)
    for r in R[half]: tab[tuple(r[k] for k in keys)].append(r[4])
    mean = {k: np.mean(v) for k, v in tab.items()}; glob = R[half, 4].mean()
    pred = np.array([mean.get(tuple(r[k] for k in keys), glob) for r in R[~half]]); y = R[~half, 4]
    return 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum(), len(mean)
for lab, keys in (("offsets to second and third", (1, 2)), ("offsets to second, third and largest", (1, 2, 3)), ("offset to second only", (1,)), ("offset to third only", (2,))):
    r2, cells = fit_score(keys); print(f"  {lab:40s} held-out share of variance explained {100*r2:5.1f}% ({cells} cells)")
y = R[~half, 4]; print(f"  signed slope on scored lists: median {np.median(y):.2f}, mean {y.mean():.2f}, sd {y.std():.2f}")

# look 7b: |slope| by the kept number's value relative to the third, and sign agreement of the offsets table
val = np.array([X[int(r[0]), int(O[int(r[0]), 1] + r[1])] for r in R]); third = X[R[:, 0].astype(int), O[R[:, 0].astype(int), 2]]
ratio = val / third; a = np.abs(R[:, 4])
for lo, hi in ((0, 0.5), (0.5, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 0.95), (0.95, 1.0)):
    m = (ratio >= lo) & (ratio < hi)
    if m.any(): print(f"  kept number at {lo:.2f}-{hi:.2f} of the third: n {m.sum():5d}, |slope| median {np.median(a[m]):6.2f}, 90th percentile {np.percentile(a[m], 90):7.2f}")

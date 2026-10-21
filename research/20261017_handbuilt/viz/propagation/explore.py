"""First look at the numbers a visual of Hilton's propagation would show: the largest so far, the store against it by
age, the leftovers of two store rules, and how joint the final scores are."""
import numpy as np
from scipy.stats import norm
from algzoo.core import example
wi, wh, wo = (np.asarray(p, float) for p in example())
print("store self-weight", wh[2, 2], "store from detector", wh[2, 4], "detector from store", wh[4, 2], "inputs", wi[2], wi[4])
rng = np.random.default_rng(5); N = 200000; X = rng.standard_normal((N, 10))
h = np.zeros((N, 16)); HS = []
for t in range(10): h = np.maximum(h @ wh.T + X[:, t:t + 1] * wi, 0); HS.append(h)
HS = np.stack(HS, 1); L = h @ wo.T
y = np.argsort(X, 1)[:, -2]; print("accuracy", (L.argmax(1) == y).mean())
Xp = np.maximum(X, 0); M = np.maximum.accumulate(Xp, 1)                    # largest so far after step t (0 if none positive)
arg = np.zeros((N, 10), int); cur = np.full(N, -1)
for t in range(10): cur = np.where(X[:, t] > np.r_[np.zeros(N)][0] if False else (X[:, t] > (M[:, t - 1] if t else 0)), t, cur); arg[:, t] = cur
age = np.where(arg >= 0, np.arange(10)[None] - arg, -1)
print("exact vs sampled P(M_t <= a), t=5, a=0.5,1,1.5:", [round(norm.cdf(a) ** 5, 4) for a in (0.5, 1, 1.5)], [round((M[:, 4] <= a).mean(), 4) for a in (0.5, 1, 1.5)])
S = HS[:, :, 2]
print("store / M by age (median, 10-90%), M > 0.3, steps 2-10:")
for A in range(0, 9):
    sel = (age == A) & (M > 0.3); sel[:, 0] = False
    r = S[sel] / M[sel]
    print(f"  age {A}: n={sel.sum():7d} median {np.median(r):6.2f}  10% {np.quantile(r, .1):6.2f}  90% {np.quantile(r, .9):6.2f}")
# leads
corr = L[np.arange(N), y][:, None] - L
lead = np.where(np.arange(10)[None] == y[:, None], np.inf, corr); mn = lead.min(1)
print("min lead > 0", (mn > 0).mean(), "median min lead", np.median(mn))
marg = [(lead[:, j][np.isfinite(lead[:, j])] > 0).mean() for j in range(10)]
# independent leads: product over the nine leads of each list's... use per-competitor-rank marginals
rk = np.argsort(-X, 1)                                                      # positions by rank
lead_by_rank = np.stack([np.where(rk[:, k] == y, np.inf, lead[np.arange(N), rk[:, k]]) for k in range(10)], 1)
p_rank = [(lead_by_rank[:, k] > 0).mean() for k in range(10) if k != 1]
print("P(lead over rank k > 0):", np.round(p_rank, 4), "product", np.prod(p_rank))
ry, r1 = L[np.arange(N), y], L[np.arange(N), rk[:, 0]]
print("corr(right score, largest's score)", np.corrcoef(ry, r1)[0, 1])
perm = rng.permutation(N); print("right beats largest: real", (ry > r1).mean(), "scores shuffled independently", (ry > r1[perm]).mean())

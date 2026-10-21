"""Near ties between a new number x_t and the old largest P (at r): which one the trained network names as second is set
by the final score margin s_r - s_t. Can one fixed rule predict that margin from two signals at step t, the cancellers'
summed activity C and the detector D (neuron 4)? Everything is divided by P (the network has no biases, so scores
scale with the input). Lists: |x_t / P - 1| < 0.05, third-largest below 0.8 P, no new largest after t."""
import numpy as np
from algzoo.core import example
wi, wh, wo = (np.asarray(w, dtype=float) for w in example()); wi = wi.reshape(-1)
rng = np.random.default_rng(40); X = rng.standard_normal((3_000_000, 10)); B = len(X)
m1p = np.stack([np.max(np.c_[np.zeros((B, 1)), X[:, :t]], 1) for t in range(10)], 1)
rows = []
for t in range(2, 10):
    P = m1p[:, t]; r = X[:, :t].argmax(1); ratio = X[:, t] / np.where(P > 0, P, np.nan)
    rest = X.copy(); rest[np.arange(B), r] = -9; rest[:, t] = -9
    m = (P > 0.2) & (np.abs(ratio - 1) < 0.05) & (rest.max(1) < 0.8 * P) & (X[:, t + 1:] < 0.8 * P[:, None]).all(1)
    rows.append((t, np.flatnonzero(m), r))
idx = np.concatenate([i for _, i, _ in rows]); T = np.concatenate([np.full(len(i), t) for t, i, _ in rows])
R = np.concatenate([r[i] for _, i, r in rows]); Xs = X[idx]; n = len(idx)
h = np.zeros((n, 16)); Ct = np.zeros(n); Dt = np.zeros(n)
for t in range(10):
    h = np.maximum(h @ wh.T + Xs[:, t:t + 1] * wi, 0)
    Ct[T == t] = h[T == t][:, [1, 6, 7, 8]].sum(1); Dt[T == t] = h[T == t, 4]
s = h @ wo.T; P = Xs[np.arange(n), R]; xt = Xs[np.arange(n), T]
marg = (s[np.arange(n), R] - s[np.arange(n), T]) / P; c, d, u = Ct / P, Dt / P, xt / P - 1
truth = u > 0                                   # old largest is second iff x_t is bigger
print(f"{n} near-tie lists; trained network picks correctly between the two: {((marg > 0) == truth).mean():.3f}")
def fit(A, y, m):
    co = np.linalg.lstsq(A[m], y[m], rcond=None)[0]; p = A @ co
    return co, 1 - ((y - p)[m] ** 2).mean() / y[m].var(), p
A = np.c_[c, d, np.ones(n)]
co, r2, p = fit(A, marg, np.ones(n, bool))
print(f"one rule for all steps and positions: margin/P = {co[0]:.2f} C/P + {co[1]:.2f} D/P + {co[2]:.2f}  R2 {r2:.2f}  "
      f"sign agrees with the network {((p > 0) == (marg > 0)).mean():.3f}")
co2, r22, p2 = fit(np.c_[u, np.ones(n)], marg, np.ones(n, bool))
print(f"baseline, the true difference only: margin/P = {co2[0]:.1f} (x_t/P - 1) + {co2[1]:.2f}  R2 {r22:.2f}  sign agrees {((p2 > 0) == (marg > 0)).mean():.3f}")
print("per step t (rule fitted to all, then per step): lists, R2 of the shared rule, R2 own rule, own coefficients C, D, const")
for t in range(2, 10):
    m = T == t
    if m.sum() < 200: continue
    r2s = 1 - ((marg - p)[m] ** 2).mean() / marg[m].var()
    cot, r2t, _ = fit(A, marg, m)
    print(f"  t={t}: {m.sum():6d}  shared {r2s:5.2f}  own {r2t:5.2f}   {cot[0]:6.2f} {cot[1]:6.2f} {cot[2]:6.2f}")
print("per gap t - r (steps between old largest and new number), shared rule R2 and own coefficients")
for g in range(1, 9):
    m = (T - R) == g
    if m.sum() < 200: continue
    r2s = 1 - ((marg - p)[m] ** 2).mean() / marg[m].var(); cot, r2t, _ = fit(A, marg, m)
    print(f"  gap {g}: {m.sum():6d}  shared {r2s:5.2f}  own {r2t:5.2f}   {cot[0]:6.2f} {cot[1]:6.2f} {cot[2]:6.2f}")

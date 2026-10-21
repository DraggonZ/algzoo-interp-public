"""Step 3f: rules for neuron 15 (and 3, 14) with sizes. Score = c + sum over the k largest positive numbers so far
(k = 1..3) of [ value_k * W_k(age_k) + U_k(age_k) ] (+ x_t * w). Off when score <= 0. Forms:
  V   : value terms only, ranks 1-3 (33 numbers + constant + current number)
  V+U : value terms and per-age constants (66 + 2)
  V1+U: value terms of the largest only, per-age constants of all three ranks
Fitted by logistic regression on the network's own on/off (seed 32), tested alone, closed loop, seed 31."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from swlib import *

X = lists(TEST_SEED, 200_000); K = kinds(X)
S0, P0, H0 = run(X); base = score(X, S0, S0, K); true_off = P0[:, :, MEM] <= 0
Xf = lists(FIT_SEED, 200_000); _, Pf, _ = run(Xf); off_f = Pf[:, :, MEM] <= 0
ag, _ = order_keys(X); agf, _ = order_keys(Xf)


def topvals(X):
    n = len(X); v = np.zeros((n, 10, 3))
    for t in range(10):
        s = -np.sort(-np.maximum(X[:, :t + 1], 0), 1)[:, :3]; v[:, t, :s.shape[1]] = s
    return v


vt, vf = topvals(X), topvals(Xf)
oh = lambda a: (a[..., None] == np.arange(-1, 10)).astype(np.float32)


def design(ag, v, X, form):
    cols = []
    for k in range(3):
        if "V" in form and (k == 0 or "V1" not in form): cols.append(oh(ag[..., k]) * v[..., k:k + 1])
        if "U" in form: cols.append(oh(ag[..., k]))
    cols.append(X[..., None].astype(np.float32))
    return np.concatenate(cols, -1)


acc = lambda S: 100 * (score(X, S0, S)["all"][0] - base["all"][0])
idx = np.random.default_rng(0).choice(len(Xf), 60_000, replace=False)
for n in [15, 3, 14]:
    j = MEM.index(n); print(f"== neuron {n}")
    for form in ["V", "V1+U", "V+U"]:
        D = design(agf[idx], vf[idx], Xf[idx], form)[:, 1:]; D = D.reshape(-1, D.shape[-1]); y = off_f[idx, 1:, j].ravel()
        used = D.any(0); lr = LogisticRegression(C=10.0, max_iter=3000).fit(D[:, used], y)
        Dt = design(ag, vt, X, form)[..., used]; pred = np.zeros((len(X), 10, 8), bool)
        pred[:, :, j] = lr.decision_function(Dt.reshape(-1, used.sum())).reshape(len(X), 10) > 0
        pred[:, 0, j] = X[:, 0] * WIM[j] <= 0
        S, offs = machine(X, lambda t, c: pred[:, t], use=[n])
        print(f"   {form:5s} fitted numbers {used.sum() + 1:4d}  agreement {np.mean(pred[:, :, j] == true_off[:, :, j]):.3f}  alone {acc(S):+.2f}")
        if n == 15 and form == "V+U": np.save("sizes15_coef.npy", np.r_[lr.intercept_, lr.coef_[0]])

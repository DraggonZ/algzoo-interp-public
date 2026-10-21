"""Step 3d: compact order rules for neurons 3, 14 and 15 (the ones the hand rules do not cover well).
Each is "off when a score is above zero", the score a sum of per-age values: logistic fits (to the network's own
on/off, seed 32, 400,000 samples of list and step with step >= 1) on one-hot codes of
  A : age of the largest (11 values) + constant
  A+B: + age of the second (11)                      (additive: one wave per rank)
  A+B+C: + age of the third (11)
  A+B+C+S: + step (9)
  AxB: one value per (age of largest, age of second) pair, + constant
  AxB+C: + age of the third
Fitted numbers = number of one-hot columns that occur + 1. Tested alone (others on their own sign), closed loop."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from swlib import *

X = lists(TEST_SEED, 200_000); K = kinds(X)
S0, P0, H0 = run(X); base = score(X, S0, S0, K); true_off = P0[:, :, MEM] <= 0
Xf = lists(FIT_SEED, 200_000); _, Pf, _ = run(Xf); off_f = Pf[:, :, MEM] <= 0
ag, _ = order_keys(X); agf, _ = order_keys(Xf)


def onehot(v, k):
    return (v[..., None] == np.arange(k)).astype(np.float32)


def design(ag, form):
    a1, a2, a3 = ag[..., 0] + 1, ag[..., 1] + 1, ag[..., 2] + 1; T = np.broadcast_to(np.arange(10), a1.shape)
    parts = {"A": onehot(a1, 11), "B": onehot(a2, 11), "C": onehot(a3, 11), "S": onehot(T, 10)[..., 1:],
             "AxB": onehot(a1 * 11 + a2, 121)}
    keys = ["AxB"] + (["C"] if "C" in form else []) if form.startswith("AxB") else form.split("+")
    return np.concatenate([parts[k] for k in keys], -1)


rng = np.random.default_rng(0); idx = rng.choice(len(Xf), 40_000, replace=False)
acc = lambda S: 100 * (score(X, S0, S)["all"][0] - base["all"][0])
for n in [15, 3, 14]:
    j = MEM.index(n); print(f"== neuron {n}")
    for form in ["A", "A+B", "A+B+C", "A+B+C+S", "AxB", "AxB+C"]:
        D = design(agf[idx], form)[:, 1:]; D = D.reshape(-1, D.shape[-1]); y = off_f[idx, 1:, j].ravel()
        used = D.any(0)
        lr = LogisticRegression(C=10.0, max_iter=2000).fit(D[:, used], y)
        Dt = design(ag, form); pred = np.zeros((len(X), 10, 8), bool)
        pred[:, :, j] = lr.decision_function(Dt[..., used].reshape(-1, used.sum())).reshape(len(X), 10) > 0
        pred[:, 0, j] = X[:, 0] * WIM[j] <= 0
        S, offs = machine(X, lambda t, c: pred[:, t], use=[n])
        print(f"   {form:8s} fitted numbers {used.sum() + 1:4d}  agreement {np.mean(pred[:, :, j] == true_off[:, :, j]):.3f}  alone {acc(S):+.2f}")

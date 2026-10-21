"""Checks. (1) Forcing the gate (neuron 5) off one step after a later second-largest, random lists.
(2) Lists whose second-largest is an overtaken record with the third right after it: trained and twin accuracy.
(3) Conceptual misreadings of the trained network's cancelling neurons (1, 6, 7, 8) next to its scaled true values:
formulas fitted by least squares on pre-activations (fit lists), substituted for the group, scored as output error in
units of the decision margin (as error_vs_margin.py) and accuracy.
  account      - our account's quantities (input, largest, second-largest, the largest's timer wave);
  wave from m2 - the timer wave's height taken from the second-largest instead of the largest;
  no wave      - the timer wave left out (input and levels only);
  wrong angle  - the wave at 20 degrees per step more than the account's angle."""
import numpy as np
import simlib
from simlib import lists, run, task_vars
from sim_groups import account_features, ANG
from algzoo.handbuilt_v18 import build_v18

X = lists(1313, 200_000); y = X.argsort(1)[:, -2]; B = len(X)
p1, p2 = X.argmax(1), y
# (1) gate off
sel = (p2 > p1) & (p2 <= 8) & (X[np.arange(B), p2] > 0)
h = np.zeros((B, 16))
for t in range(10):
    h = np.maximum(h @ simlib.WH.T + X[:, t:t + 1] * simlib.WI, 0)
    m = sel & (p2 + 1 == t); h[m, 5] = 0
L = run(X)[0]
print(f"(1) gate off one step after a later positive second-largest ({sel.sum()} lists): accuracy {np.mean(L[sel].argmax(1) == y[sel]):.3f} -> {np.mean((h[sel] @ simlib.WO.T).argmax(1) == y[sel]):.3f}")
# (2) overtaken record as second, third right after it
m1p = np.stack([np.max(np.c_[np.full(B, -np.inf), X[:, :t]], 1) for t in range(10)], 1)
was_rec = X[np.arange(B), p2] > m1p[np.arange(B), p2]
p3 = X.argsort(1)[:, -3]; sel2 = (p2 < p1) & was_rec & (p3 == p2 + 1)
P18 = tuple(np.asarray(p, float) for p in build_v18().params)
L18 = run(X, params=P18)[0]
print(f"(2) second an overtaken record, third right after it ({sel2.sum()} lists): trained {np.mean(L[sel2].argmax(1) == y[sel2]):.3f}, twin {np.mean(L18[sel2].argmax(1) == y[sel2]):.3f}")


# (3) misreadings
def feats(Xs, n, kind):
    if kind == "account":
        return account_features(Xs, n)
    V = task_vars(Xs); x, m1, m2, age = V["x"], V["m1"], V["m2"], V["age"]; live = age >= 0
    th = np.radians(ANG[n] + (20 if kind == "wrong angle" else 0)); hgt = m2 if kind == "wave from m2" else m1
    F = [x, m1, m2]
    if kind != "no wave":
        F += [np.where(live, hgt * np.cos(age * th), 0), np.where(live, hgt * np.sin(age * th), 0)]
    return np.stack(F, -1)


Xf = lists(1314, 100_000); _, PREf, _ = run(Xf)
Xs = X[:50_000]; Ls = L[:50_000]; ys = y[:50_000]
srt = np.sort(Ls, 1); margin = srt[:, -1] - srt[:, -2]
CANC = [1, 6, 7, 8]
for kind in ("account", "wave from m2", "no wave", "wrong angle"):
    sims = {}
    for n in CANC:
        Ff = feats(Xf, n, kind); co = np.linalg.lstsq(Ff.reshape(-1, Ff.shape[-1]), PREf[:, :, n].reshape(-1), rcond=None)[0]
        sims[n] = np.maximum(feats(Xs, n, kind) @ co, 0)
    Lsub = run(Xs, sims)[0]
    err = np.median(np.abs(Lsub - Ls).max(1) / np.maximum(margin, 1e-12))
    print(f"(3) cancellers from '{kind}': output error {err:.2f} margins, accuracy {np.mean(Lsub.argmax(1) == ys):.3f}", flush=True)
for e in (0.02, 0.05, 0.1, 0.2):
    h = np.zeros((len(Xs), 16)); f = np.ones(16); f[CANC] = 1 + e
    for t in range(10):
        h = np.maximum(h @ simlib.WH.T + Xs[:, t:t + 1] * simlib.WI, 0) * f
    Lsub = h @ simlib.WO.T; err = np.median(np.abs(Lsub - Ls).max(1) / np.maximum(margin, 1e-12))
    print(f"(3) cancellers' true values scaled {1 + e:.2f}: output error {err:.2f} margins, accuracy {np.mean(Lsub.argmax(1) == ys):.3f}", flush=True)

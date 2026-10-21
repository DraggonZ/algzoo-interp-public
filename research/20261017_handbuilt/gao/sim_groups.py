"""Simulation score per neuron group (Gao's test). For each group, its activations at every step are replaced by values
simulated from the input list, the rest of the trained network runs normally, and the output's KL from the trained
output and the accuracy are recorded. Three simulators per group:
  mean     - the group's average at each step (knows nothing about the list),
  generic  - per step, a linear function of all inputs so far, then ReLU (no understanding; 55 coefficients per neuron),
  account  - a ReLU of the quantities our account names for that group, with one set of coefficients for all steps.
Coefficients are fitted to the pre-activations on lists of seed 101 and scored on seed 202 (200,000 lists each)."""
import numpy as np
from simlib import GROUPS, lists, run, task_vars, score, fit_relu

ANG = {1: 56, 6: 46, 7: 39, 8: 30}   # timer angles from cancellers/arc_canceller_code.log


def lag(a, k=1):
    out = np.zeros_like(a); out[:, k:] = a[:, :-k]; return out


def account_features(X, n):
    V = task_vars(X); B = len(X)
    x, m1, m2, m1in, jump, age = V["x"], V["m1"], V["m2"], V["m1in"], V["jump"], V["age"]
    if n in (2, 4):          # the largest so far, and the jump that makes a new one
        F = [m1in, m1, x, jump]
    elif n in ANG:           # cancellers: gain x (switch-off point - input) + timer wave of the current largest
        th = np.radians(ANG[n]); live = age >= 0
        F = [x, m1, m2, np.where(live, m1 * np.cos(age * th), 0), np.where(live, m1 * np.sin(age * th), 0)]
    elif n == 5:             # keep gate: the previous input against the second-largest before it
        F = [lag(x), lag(m2), lag(m1), x, m1, m2]
    elif n == 9:             # late gate: on one step after a new largest
        F = [lag(jump), lag(x), x, m1, m2]
    else:                    # memory: kept numbers by age (a time-invariant shift register), plus levels
        kept = (x > 0) & (x >= 0.75 * m2)
        kv = np.where(kept, x, 0)
        F = [lag(kv, a) for a in range(1, 10)] + [kv, m1in, m1, m2]
    return np.stack(F, -1)


def generic_features(X):
    B = len(X); F = np.zeros((B, 10, 10 * 10))
    for t in range(10):
        F[:, t, t * 10:t * 10 + t + 1] = X[:, :t + 1]
    return F


if __name__ == "__main__":
    Xf, Xs = lists(101), lists(202)
    _, Pf, Hf = run(Xf); Ls, Ps, Hs = run(Xs)
    Gf, Gs = generic_features(Xf), generic_features(Xs)
    sims = {}
    for n in range(16):
        sims[("mean", n)] = np.broadcast_to(Hf[:, :, n].mean(0), (len(Xs), 10))
        sims[("generic", n)] = fit_relu(Gf, Pf[:, :, n], Gs)[0]
        sims[("account", n)] = fit_relu(account_features(Xf, n), Pf[:, :, n], account_features(Xs, n))[0]
    rows = list(GROUPS.items()) + [("all 16", list(range(16)))]
    print(f"{'group':16s} {'simulator':8s} {'KL (nats)':>9s} {'accuracy':>8s} {'agree':>6s} {'R2 of values':>12s}")
    print(f"{'none':16s} {'':8s} {0:9.3f} {(Ls.argmax(1) == Xs.argsort(1)[:, -2]).mean():8.3f} {1:6.3f}")
    for name, ns in rows:
        for kind in ("mean", "generic", "account"):
            sub = {n: sims[(kind, n)] for n in ns}
            s = score(Xs, Ls, run(Xs, sub)[0])
            r2 = 1 - sum(((Hs[:, :, n] - sims[(kind, n)]) ** 2).sum() for n in ns) / sum(((Hs[:, :, n] - Hs[:, :, n].mean()) ** 2).sum() for n in ns)
            print(f"{name:16s} {kind:8s} {s['kl']:9.3f} {s['acc']:8.3f} {s['agree']:6.3f} {r2:12.3f}")

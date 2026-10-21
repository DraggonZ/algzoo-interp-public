"""Leaner hiding-group rules: fewest constants that keep every hiding neuron at least 3x better than the account.
Base L1: account's 5 + store/detector rule (2) + previous number + late-gate proxy (previous jump) + 4 feedback = 13.
Additions: kept numbers as decaying waves at the neuron's own timer angle (2: sum over ages of kept value x rho^age
x cos/sin(age theta), rho 0.9), memory-mode waves of the largest (7), kept-number lags (9)."""
import numpy as np
from harness import *  # noqa
import step3_recursive as s3
from step3_recursive import account_features
import step4_ablate as s4
sys.path.insert(0, "/data/output/research/algzoo/20261017_handbuilt/gao")
from sim_groups import ANG

kw_cache = {}


def keptwave(X, n, rho=0.9):
    key = (id(X), n)
    if key not in kw_cache:
        V = task_vars(X); x, m2 = V["x"], V["m2"]
        kv = np.where((x > 0) & (x >= 0.75 * m2), x, 0); th = np.radians(ANG[n])
        c = sum(lag(kv, a) * rho ** a * np.cos(a * th) for a in range(1, 10))
        s = sum(lag(kv, a) * rho ** a * np.sin(a * th) for a in range(1, 10))
        kw_cache[key] = [c, s]
    return kw_cache[key]


def make(use, kept_wave=False):
    def f(X, n):
        V, P = s4.parts(X)
        F = [account_features(X, n)[..., k] for k in range(5)]
        for k in use:
            F += P[k]
        if kept_wave:
            F += keptwave(X, n)
        return np.stack(F, -1)
    return f


if __name__ == "__main__":
    variants = {"L1: account + store/detector rule + previous number + late gate": (["sd", "xprev", "lategate"], False),
                "L1 + kept numbers as one wave": (["sd", "xprev", "lategate"], True),
                "L1 + memory-mode waves": (["sd", "xprev", "lategate", "modes"], False),
                "L1 + kept-number lags": (["sd", "xprev", "lategate", "kept"], False)}
    for lab, (use, kwv) in variants.items():
        s3.fit_group(make(use, kwv), lab)

"""Which ingredients of rule B matter, and what else helps neuron 7? Recursive group rule as in step3; each variant
drops or adds one ingredient for all four neurons. Additions: the late gate's proxy (previous step's jump), the
previous number, and the keep gate's proxy (previous number kept: x >= 0.75 m2)."""
import numpy as np
from harness import *  # noqa
import step3_recursive as s3
from step3_recursive import account_features, sd_rule, MODES

cache = {}


def parts(X):
    key = id(X)
    if key in cache:
        return cache[key]
    V = task_vars(X); x, m1, m2, age, jump = V["x"], V["m1"], V["m2"], V["age"], V["jump"]; live = age >= 0
    kept = (x > 0) & (x >= 0.75 * m2); kv = np.where(kept, x, 0)
    sd = sd_rule(X)
    P = {"sd": [sd[..., 0], sd[..., 1]], "kept": [lag(kv, a) for a in range(1, 10)], "modes": [],
         "lategate": [lag(jump)], "xprev": [lag(x)], "keptprev": [lag(kv)], "m1fade": []}
    for deg, rho in MODES:
        amp = np.where(live, m1 * rho ** np.maximum(age, 0), 0); th = np.radians(deg)
        P["modes"] += [amp * np.cos(age * th)] + ([amp * np.sin(age * th)] if deg != 180 else [])
    P["m1fade"] = [np.where(live, m1 * 0.969 ** np.maximum(age, 0), 0)]
    cache[key] = (V, P)
    return V, P


def make(use):
    def f(X, n):
        V, P = parts(X)
        F = [account_features(X, n)[..., k] for k in range(5)]
        for k in use:
            F += P[k]
        return np.stack(F, -1)
    return f


if __name__ == "__main__":
    import io, contextlib
    variants = {"B (sd, kept, modes)": ["sd", "kept", "modes"],
                "B without store/detector rule": ["kept", "modes"],
                "B without kept-number lags": ["sd", "modes"],
                "B without memory-mode waves": ["sd", "kept"],
                "B + late-gate proxy": ["sd", "kept", "modes", "lategate"],
                "B + previous number": ["sd", "kept", "modes", "xprev"],
                "B + late gate + previous number": ["sd", "kept", "modes", "lategate", "xprev"]}
    for lab, use in variants.items():
        s3.fit_group(make(use), lab)

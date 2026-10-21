"""Shared pieces for the gate study (neurons 5 and 9 of ARC's 16-neuron second-argmax RNN).

Conventions (as in ../gao/simlib.py):
- Lists are ten standard-normal numbers, positions 0..9.
- run() returns pre-activations P and activations H with H[:, t] the state after input t.
- Neuron 5's value "one step after x_t" is H[:, t + 1, 5]; it judges the number x_t.
- The largest and second-largest "so far" are taken with zeros padded (the network has no biases, so a list that is
  all negative so far looks to it like a largest of 0). m1[t], m2[t] are the largest and second-largest BEFORE input t.
"""
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
import numpy as np
from algzoo.core import example, zoo, load

WI, WH, WO = (np.asarray(w, dtype=float) for w in example())
WI = WI.reshape(-1)
MEM = [0, 3, 10, 11, 12, 13, 14, 15]
CANC = [1, 6, 7, 8]


def lists(seed, n=200_000):
    return np.random.default_rng(seed).standard_normal((n, 10))


def run(X, params=None, sub=None, hook=None):
    """Run a network. sub: {neuron: array (B, 10)} forced after each step's update.
    hook(t, h_prev, pre, h) -> h may change the new state in place (interventions that need the run's own values).
    Returns final scores (B, 10), pre-activations (B, 10, n) and activations (B, 10, n)."""
    wi, wh, wo = params if params is not None else (WI, WH, WO)
    wi = np.asarray(wi, float).reshape(-1)
    B = len(X); h = np.zeros((B, len(wi))); P, H = [], []
    for t in range(10):
        pre = h @ wh.T + X[:, t:t + 1] * wi
        hn = np.maximum(pre, 0)
        if sub:
            for n, v in sub.items():
                hn[:, n] = v[:, t]
        if hook is not None:
            hn = hook(t, h, pre, hn)
        h = hn
        P.append(pre); H.append(h.copy())
    return h @ np.asarray(wo, float).T, np.stack(P, 1), np.stack(H, 1)


def answer(X):
    return X.argsort(1)[:, -2]


def task_vars(X):
    """Per list and step t: x, m1, m2 (largest, second-largest before input t, zero padded), a1 and a2 (their ages in
    steps at time t, -1 if the value is the zero pad), new (x_t is a record, x > m1), p (the previous largest at the last
    record, i.e. the value the current largest overtook)."""
    B = len(X)
    V = {k: np.zeros((B, 10)) for k in ("x", "m1", "m2", "a1", "a2", "new", "p", "m1in", "m2in", "a1in", "a2in")}
    m1 = np.zeros(B); m2 = np.zeros(B); i1 = np.full(B, -1); i2 = np.full(B, -1); p = np.zeros(B)
    for t in range(10):
        x = X[:, t]
        V["x"][:, t] = x; V["m1"][:, t] = m1; V["m2"][:, t] = m2; V["p"][:, t] = p
        V["a1"][:, t] = np.where(i1 >= 0, t - i1, -1); V["a2"][:, t] = np.where(i2 >= 0, t - i2, -1)
        new = x > m1; sec = ~new & (x > m2)
        V["new"][:, t] = new
        p = np.where(new, m1, p)
        m2n = np.where(new, m1, np.where(sec, x, m2)); i2 = np.where(new, i1, np.where(sec, t, i2))
        m1 = np.where(new, x, m1); i1 = np.where(new, t, i1); m2 = m2n
        V["m1in"][:, t] = m1; V["m2in"][:, t] = m2
        V["a1in"][:, t] = np.where(i1 >= 0, t - i1, -1); V["a2in"][:, t] = np.where(i2 >= 0, t - i2, -1)
    return V


def kinds(X):
    """Boolean masks for kinds of list, by where the largest (L) and second-largest (S) sit."""
    o = X.argsort(1); iL, iS = o[:, -1], o[:, -2]
    pm = np.maximum.accumulate(np.c_[np.full(len(X), -np.inf), X], 1)[:, :-1]
    rec = X > pm
    K = {"all": np.ones(len(X), bool),
         "S before L (revealed)": iS < iL,
         "S after L (kept)": iS > iL,
         "S at 8": iS == 8, "S at 9": iS == 9, "L at 9": iL == 9, "L at 8": iL == 8,
         "record at 8": rec[:, 8], "record at 9": rec[:, 9],
         "S right after L": iS == iL + 1,
         "S before L, L at 9": (iS < iL) & (iL == 9),
         "S after L, S at 8": (iS > iL) & (iS == 8),
         "S after L, S at 9": (iS > iL) & (iS == 9),
         }
    return K


def acc_by_kind(X, L, keys=None):
    y = answer(X); ok = L.argmax(1) == y; K = kinds(X)
    keys = keys or list(K)
    return {k: (ok[K[k]].mean(), K[k].sum()) for k in keys}


def fmt_kinds(X, L, keys=None, base=None):
    r = acc_by_kind(X, L, keys)
    out = []
    for k, (a, n) in r.items():
        s = f"{k}: {100 * a:.1f}%"
        if base is not None:
            s += f" (base {100 * base[k][0]:.1f})"
        out.append(s)
    return " | ".join(out)


def other_models():
    """The 21 other trained 16-neuron networks, loaded as in ../bridge/second_comparator_across.py."""
    models = {}
    for s in range(5):
        models[f"zoo{s}"] = zoo(16, 10, s)
    for f in sorted(Path("/data/output/research/algzoo/20261016_run3/models").glob("2nd_argmax_16_10_*_1*.pth")):
        models["new" + f.stem.split("_")[-1]] = load(str(f))
    return {k: tuple(np.asarray(p, float) for p in w) for k, w in models.items()}

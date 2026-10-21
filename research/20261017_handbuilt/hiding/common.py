"""Shared pieces for the hiding-group study (trained network, neurons 1, 6, 7, 8)."""
import numpy as np
from algzoo.core import example

WI, WH, WO = (np.asarray(w, float) for w in example())
WI = WI.reshape(-1)
C = [1, 6, 7, 8]
GROUPS = {"hiding": [1, 6, 7, 8], "store": [2], "detector": [4], "keep gate": [5], "late gate": [9],
          "memory": [0, 3, 10, 11, 12, 13, 14, 15]}


def lists(seed, n):
    return np.random.default_rng(seed).standard_normal((n, 10))


def run(X, params=None, hook=None):
    """Run a network. hook(t, pre, h) may return a modified h. Returns final scores, pre-activations and states (B,10,16)."""
    wi, wh, wo = params if params is not None else (WI, WH, WO)
    wi = np.asarray(wi, float).reshape(-1)
    B = len(X); h = np.zeros((B, len(wi))); P, H = [], []
    for t in range(10):
        pre = h @ wh.T + X[:, t:t + 1] * wi; hn = np.maximum(pre, 0)
        if hook is not None:
            hn = hook(t, pre, hn, h)
        h = hn; P.append(pre); H.append(h.copy())
    return h @ wo.T, np.stack(P, 1), np.stack(H, 1)


def task(X):
    """Per list and step t: p = previous largest before the current record (0 floor), m1 = largest up to and including t,
    r = position of the current largest, k = t - r, rec = whether t is a record. Largest counted with a floor of 0 as in
    the earlier fits (lists that start negative have m1 = 0 until a positive number)."""
    B = len(X); m1 = np.zeros(B); p = np.zeros(B); r = np.full(B, -1)
    M1 = np.zeros((B, 10)); P = np.zeros((B, 10)); R = np.zeros((B, 10), int); REC = np.zeros((B, 10), bool)
    for t in range(10):
        x = X[:, t]; new = x > m1
        p = np.where(new, m1, p); m1 = np.where(new, x, m1); r = np.where(new, t, r)
        M1[:, t] = m1; P[:, t] = p; R[:, t] = r; REC[:, t] = new
    return {"m1": M1, "p": P, "r": R, "k": np.arange(10)[None] - R, "rec": REC}


def kinds(X):
    """Kinds of list used across the project (they overlap)."""
    o = np.argsort(-X, 1); lar, sec, thi = o[:, 0], o[:, 1], o[:, 2]
    srt = -np.sort(-X, 1)
    pm = np.maximum.accumulate(np.c_[np.full(len(X), -np.inf), X], 1)[:, :-1]; rec = X > pm
    sec_rec = rec[np.arange(len(X)), sec]
    close = srt[:, 2] / np.where(srt[:, 1] != 0, srt[:, 1], 1) > 0.9
    first_rec_after = np.full(len(X), -1)
    for t in range(9, -1, -1):
        first_rec_after = np.where(rec[:, t] & (t > sec), t, first_rec_after)
    erase = sec_rec & (thi == sec + 1) & (first_rec_after > sec + 1) & (lar == first_rec_after)
    return {
        "all": np.ones(len(X), bool),
        "second before largest": sec < lar,
        "second after largest": sec > lar,
        "second was overtaken record": sec_rec & (sec < lar),
        "overtaken by under 5%": sec_rec & (sec < lar) & (srt[:, 0] < 1.05 * srt[:, 1]),
        "third within 10% of second": close,
        "erasure lists (third right after the record, then overtaken)": erase,
        "at most one positive": (X > 0).sum(1) <= 1,
    }

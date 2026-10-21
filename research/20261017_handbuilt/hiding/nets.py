"""The 22 trained networks used across the project (as bridge/second_comparator_across.py), and the hiding neurons by
the 85% rule: off at 85% or more of records (steps 1-9) and on at half or more of other steps (steps 2-9)."""
from pathlib import Path
import numpy as np
from algzoo.core import example, zoo, load


def models():
    m = {"ARC": example()}
    for s in range(5): m[f"zoo{s}"] = zoo(16, 10, s)
    for f in sorted(Path("/data/output/research/algzoo/20261016_run3/models").glob("2nd_argmax_16_10_*_1*.pth")):
        m["new" + f.stem.split("_")[-1]] = load(str(f))
    return {k: tuple(np.asarray(p, float) for p in v) for k, v in m.items()}


def hiding(params, X, HS=None):
    wi, wh, wo = params; wi = wi.reshape(-1); N = len(X)
    if HS is None:
        h = np.zeros((N, len(wi))); HS = []
        for t in range(10): h = np.maximum(h @ wh.T + X[:, t:t + 1] * wi, 0); HS.append(h)
        HS = np.stack(HS, 1)
    pm = np.maximum.accumulate(np.c_[np.full(N, -np.inf), X], 1)[:, :-1]; rec = X > pm
    ar = np.arange(10)[None, :]; recm = rec & (ar >= 1); oth = ~rec & (ar >= 2)
    return [n for n in range(len(wi)) if (HS[:, :, n][recm] <= 1e-9).mean() >= 0.85 and (HS[:, :, n][oth] > 1e-9).mean() >= 0.5]

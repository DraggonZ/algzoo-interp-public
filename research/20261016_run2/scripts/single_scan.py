"""Is failure under one added low input a threshold in its value? Deterministic scan of x in (-3, m2) for fixed
(p1, p2, m1, m2, q), all other inputs zero. Prints the x-intervals where the network is wrong."""
import numpy as np
import torch as th

from algzoo.bench import load_bench
from algzoo.core import example, forward

names, acc, models = load_bench(("mult",))
get = lambda n: example() if n == "example" else models[list(names).index(n)]
xs = np.linspace(-3, 1.0, 4001)


def wrong_intervals(params, p1, p2, m1, m2, q):
    X = np.zeros((len(xs), 10))
    X[:, p1], X[:, p2] = m1, m2
    grid = xs[xs < m2]
    X = X[: len(grid)]
    X[:, q] = grid
    ok = (forward(params, th.as_tensor(X)).argmax(-1) == p2).numpy()
    out, start = [], None
    for i, o in enumerate(ok):
        if not o and start is None:
            start = grid[i]
        if o and start is not None:
            out.append((round(start, 3), round(grid[i - 1], 3)))
            start = None
    if start is not None:
        out.append((round(start, 3), round(grid[-1], 3)))
    return out


rng = np.random.default_rng(0)
for nm in ["example", "mult0.02_18", "mult0.01_14", "mult0.02_3"]:
    print(f"\n{nm}: wrong x-intervals for one added low input (m1=1.5, m2=1.0 unless noted)")
    shown = 0
    for p1 in range(10):
        for p2 in range(10):
            if p1 == p2:
                continue
            for q in range(min(p1, p2) + 1, 10):
                if q in (p1, p2):
                    continue
                w = wrong_intervals(get(nm), p1, p2, 1.5, 1.0, q)
                full = len(w) == 1 and w[0][0] <= -2.99 and w[0][1] >= 0.99
                if w and not full and rng.random() < 0.08 and shown < 12:
                    print(f"  pair ({p1},{p2}) q={q}: {w}")
                    shown += 1
    # summary: share of (pair, q) whose wrong set is empty / a single top interval / something else
    kinds = {"none": 0, "top only": 0, "all": 0, "other": 0}
    for p1 in range(10):
        for p2 in range(10):
            if p1 == p2:
                continue
            for q in range(10):
                if q in (p1, p2):
                    continue
                w = wrong_intervals(get(nm), p1, p2, 1.5, 1.0, q)
                if not w:
                    kinds["none"] += 1
                elif len(w) == 1 and w[0][0] <= -2.99 and w[0][1] >= 0.99:
                    kinds["all"] += 1
                elif len(w) == 1 and w[0][1] >= 0.99:
                    kinds["top only"] += 1
                else:
                    kinds["other"] += 1
    print("  kinds over 720 (pair, q):", kinds)

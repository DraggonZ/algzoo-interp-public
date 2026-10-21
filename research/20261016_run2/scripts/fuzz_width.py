"""How close must one low input come to the 2nd max to be misread? Deterministic scan (top two at m1 = 1.5,
m2 = 1.0, one low input x at position q, others zero). For every (p1, p2, q): the width of the failing interval
just below m2, w = m2 - (lowest x of the top failing interval); w = 0 if the network is right up to x -> m2.
Pairs already wrong with zero background are excluded."""
import numpy as np
import torch as th

from algzoo.bench import load_bench
from algzoo.core import example, forward

xs = th.linspace(-3, 0.9999, 8001, dtype=th.float64)
names, acc, models = load_bench(("mult",))


def widths(params):
    out = []
    for p1 in range(10):
        for p2 in range(10):
            if p1 == p2:
                continue
            X0 = th.zeros(1, 10, dtype=th.float64)
            X0[0, p1], X0[0, p2] = 1.5, 1.0
            if forward(params, X0).argmax(-1).item() != p2:
                continue
            for q in range(10):
                if q in (p1, p2):
                    continue
                X = th.zeros(len(xs), 10, dtype=th.float64)
                X[:, p1], X[:, p2], X[:, q] = 1.5, 1.0, xs
                ok = (forward(params, X).argmax(-1) == p2).numpy()
                if ok[-1]:
                    out.append(0.0)
                else:
                    i = len(ok) - 1
                    while i > 0 and not ok[i - 1]:
                        i -= 1
                    out.append(1.0 - xs[i].item())
    return np.array(out)


for nm, p in [("base", example())] + [(n, models[list(names).index(n)]) for n in ["mult0.005_6", "mult0.01_14", "mult0.02_18", "mult0.02_3"]]:
    w = widths(p)
    print(f"{nm:12s} (pair, q) cases {len(w)}: share with a failing interval {np.mean(w > 0):.2f}; width median {np.median(w[w > 0]):.3f},"
          f" 90th percentile {np.quantile(w[w > 0], 0.9):.3f}; mean width {w.mean():.3f} (x relative to m2 = 1)")

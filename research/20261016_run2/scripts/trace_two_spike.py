"""Trace logits and activation patterns of a two-spike input through time (deterministic, no sampling)."""
import sys
import numpy as np
from algzoo.bench import load_bench
from algzoo.core import example

names, acc, models = load_bench(("mult",))
get = lambda n: example() if n == "example" else models[list(names).index(n)]


def trace(params, p1, p2, rho):
    w_hi, w_hh, w_oh = params
    h = np.zeros(16)
    rows = []
    for t in range(10):
        x = 1.0 if t == p1 else (rho if t == p2 else 0.0)
        h = np.maximum(w_hh @ h + w_hi * x, 0)
        l = w_oh @ h
        # padded-slot convention: after t+1 inputs, input s sits at slot s + 9 - t
        slot2 = p2 + 9 - t
        rows.append((t, "".join("1" if v > 0 else "." for v in h), np.argmax(l) - (9 - t), np.round(np.linalg.norm(h), 3),
                     np.round(l - l.max(), 2)))
    return rows

np.set_printoptions(linewidth=220, precision=2, suppress=True)
for nm in sys.argv[1].split(","):
    p1, p2, rho = int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4])
    print(f"\n{nm}  p1={p1} p2={p2} rho={rho}   columns: t, active neurons, pointer (input index), |h|, logits - max")
    for r in trace(get(nm), p1, p2, rho):
        print(*r)

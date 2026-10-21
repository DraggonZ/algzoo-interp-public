"""Idealized-network step 2: predict all 720 tie margins from about 100 simple runs.
Reference lists (background -0.46 everywhere else, largest g):
  R0(p1):    largest alone at p1                      -> centred final scores S0[p1]      (10 runs)
  R1(p1, q): largest at p1 and a 1.0 at q             -> centred final scores S1[p1, q]   (90 runs)
Vote of a 1.0 at q given the largest at p1: V[p1, q] = S1[p1, q] - S0[p1].
Additive prediction for the tie (second and third both 1.0 at p2, p3): S0[p1] + V[p1, p2] + V[p1, p3],
so margin = S[p3] - S[p2]. Compared with the exact tie margin (720 runs)."""
import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from bias_anatomy import anatomy  # noqa: E402

from algzoo.core import example, load, zoo  # noqa: E402

pl = np.array(list(itertools.permutations(range(10), 3))); r = np.arange(len(pl))


def scores(P, X):
    wi, wh, wo = P
    h = np.zeros((len(X), 16))
    for t in range(10):
        h = np.maximum(0, h @ wh.T + X[:, t:t + 1] * wi)
    s = h @ wo.T
    return s - s.mean(1, keepdims=True)


def predict(P, g=1.6, bg=-0.46):
    X0 = np.full((10, 10), bg); X0[np.arange(10), np.arange(10)] = g
    S0 = scores(P, X0)
    pairs = np.array([(p, q) for p in range(10) for q in range(10) if p != q])
    X1 = np.full((len(pairs), 10), bg); X1[np.arange(len(pairs)), pairs[:, 0]] = g; X1[np.arange(len(pairs)), pairs[:, 1]] = 1.0
    S1 = scores(P, X1); V = np.zeros((10, 10, 10))
    V[pairs[:, 0], pairs[:, 1]] = S1 - S0[pairs[:, 0]]
    S = S0[pl[:, 0]] + V[pl[:, 0], pl[:, 1]] + V[pl[:, 0], pl[:, 2]]
    return S[r, pl[:, 2]] - S[r, pl[:, 1]], S0, V


if __name__ == "__main__":
    models = {"ARC": example(), **{f"zoo{s}": zoo(16, 10, s) for s in range(5)}}
    for f in sorted(Path("/data/output/research/algzoo/20261016_run3/models").glob("2nd_argmax_16_10_*_1*.pth"))[:6]:
        models["s" + f.stem.split("_")[-1]] = load(str(f))
    for name, P in models.items():
        pred, S0, V = predict(P)
        m = anatomy(P)[0]
        before = pl[:, 1] < pl[:, 0]
        c = lambda a, b: np.corrcoef(a, b)[0, 1]
        print(f"{name:5s}: corr(predicted, exact tie margin) all {c(pred, m):+.2f} | second before largest {c(pred[before], m[before]):+.2f} | after {c(pred[~before], m[~before]):+.2f}; "
              f"rms exact {np.sqrt(np.mean(m**2)):.2f}, rms of (pred - exact) {np.sqrt(np.mean((pred-m)**2)):.2f}")

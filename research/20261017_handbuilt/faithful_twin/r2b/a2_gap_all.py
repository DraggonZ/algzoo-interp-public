"""Analysis: job 11's test (neuron carries the gap: pre-activation regressed on largest, second, third, const in cells
step 3-9 x age 1-2) applied to every neuron of a candidate (ARC order), and to ARC.
Usage: a2_gap_all.py <candidate.py | ARC>"""
import sys, importlib.util
import numpy as np
from algzoo.core import example

arg = sys.argv[1]
if arg == "ARC":
    W = [np.asarray(a, float) for a in example()]
else:
    spec = importlib.util.spec_from_file_location("c", arg); m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m); W = [np.asarray(a, float) for a in m.weights()]
rng = np.random.default_rng(7); X = rng.standard_normal((200_000, 10)); n = len(X)
h = np.zeros((n, 16)); P = []
for t in range(10):
    p = h @ W[1].T + X[:, t:t + 1] * W[0]; h = np.maximum(p, 0); P.append(p)
P = np.stack(P, 1)
age = np.zeros((n, 10), int); cur = np.zeros(n, int)
for t in range(1, 10):
    cur = np.where(X[:, t] > X[np.arange(n), cur], t, cur); age[:, t] = t - cur
for j in range(16):
    cells = []
    for t in range(3, 10):
        v = -np.sort(-X[:, :t + 1], 1)[:, :3]
        for a0 in (1, 2):
            mm = (age[:, t] == a0) & (v[:, 0] > 0); y = P[mm, t, j]
            cf, *_ = np.linalg.lstsq(np.c_[v[mm], np.ones(mm.sum())], y, rcond=None)
            gp = v[mm, 0] - v[mm, 1]; r2 = np.corrcoef(gp, y)[0, 1] ** 2 if y.std() > 1e-12 else 0.0
            cells.append((cf[0] > 0 and cf[1] < 0, r2))
    s = np.mean([c[0] for c in cells]); r = np.median([c[1] for c in cells])
    print(f"slot {j:2d}: signs {s:.2f}  R2 {r:.3f}  {'PASS' if s >= 0.75 and r >= 0.1 else ''}")

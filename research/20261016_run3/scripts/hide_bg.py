"""Is hiding a lone input forced by the task? (pre-registered in the run history, step 83)
For each trained model: spread (max - min) of centred final scores with the largest alone (1.5 at position p, the rest at
background b) divided by the spread with the largest and a second (1.5 at p, 1.0 at q, the rest at b), median over p and
over (p, q). Background 0 (the clean two-spike case) and -0.32 (typical of the eight smaller inputs)."""
import glob

import numpy as np

from algzoo.core import example, load, zoo

models = [("example", example())] + [(f"seed{s}", zoo(16, 10, s)) for s in range(5)]
models += [(f.rsplit("_", 1)[1].split(".")[0], load(f)) for f in sorted(glob.glob("/data/output/research/algzoo/20261016_run3/models/2nd_argmax_16_10_1073741824_*.pth"))]


def scores(params, X):
    w_hi, w_hh, w_oh = (np.asarray(q, np.float64) for q in params)
    h = np.zeros((X.shape[0], len(w_hi)))
    for t in range(10):
        h = np.maximum(0, h @ w_hh.T + X[:, t, None] * w_hi)
    s = h @ w_oh.T
    return s - s.mean(1, keepdims=True)


res = {}
for name, p in models:
    out = []
    for b in (0.0, -0.32):
        X1 = np.full((10, 10), b); X1[np.arange(10), np.arange(10)] = 1.5
        lone = np.median(np.ptp(scores(p, X1), 1))
        pairs = [(i, j) for i in range(10) for j in range(10) if i != j]
        X2 = np.full((len(pairs), 10), b)
        for k, (i, j) in enumerate(pairs):
            X2[k, i] = 1.5; X2[k, j] = 1.0
        two = np.median(np.ptp(scores(p, X2), 1))
        out.append(lone / two)
    res[name] = out
    print(f"{name}: lone/two spread {out[0]:.2f} at background 0, {out[1]:.2f} at -0.32", flush=True)
R = np.array(list(res.values()))
print(f"hidden (ratio <= 0.3): {np.sum(R[:, 0] <= 0.3)} of {len(R)} at background 0, {np.sum(R[:, 1] <= 0.3)} of {len(R)} at -0.32; "
      f"seeds 0 and 4 at -0.32: {res['seed0'][1]:.2f}, {res['seed4'][1]:.2f}")

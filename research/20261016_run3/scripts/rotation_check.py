"""Does every trained model's lone-input response rotate within the ages that matter (1-9)? (goal 1, all 22 models)
Lone input 1.5 at position 0, the rest at background b; response = state minus the background-only state. PCA of the
responses at ages 1-9: share of the top two directions, and the median turning angle per step in that plane."""
import glob

import numpy as np

from algzoo.core import example, load, zoo

models = [("example", example())] + [(f"seed{s}", zoo(16, 10, s)) for s in range(5)]
models += [(f.rsplit("_", 1)[1].split(".")[0], load(f)) for f in sorted(glob.glob("/data/output/research/algzoo/20261016_run3/models/2nd_argmax_16_10_1073741824_*.pth"))]


def states(params, x):
    w_hi, w_hh, _ = (np.asarray(q, np.float64) for q in params)
    h = np.zeros(len(w_hi)); H = []
    for t in range(len(x)):
        h = np.maximum(0, w_hh @ h + x[t] * w_hi); H.append(h)
    return np.array(H)


rows = []
for name, p in models:
    out = []
    for b in (0.0, -0.32):
        x = np.full(10, b); x0 = x.copy(); x[0] = 1.5
        R = (states(p, x) - states(p, x0))[1:10]          # ages 1-9
        U, S, Vt = np.linalg.svd(R - R.mean(0), full_matrices=False)
        P = (R - R.mean(0)) @ Vt[:2].T
        ang = np.degrees(np.diff(np.unwrap(np.arctan2(P[:, 1], P[:, 0]))))
        out += [float((S[:2] ** 2).sum() / (S ** 2).sum()), float(np.median(np.abs(ang)))]
    rows.append(out)
    print(f"{name}: background 0: two-direction share {out[0]:.2f}, turn {out[1]:.0f} deg/step; background -0.32: share {out[2]:.2f}, turn {out[3]:.0f} deg/step", flush=True)
R = np.array(rows)
print(f"turn at background -0.32: {R[:, 3].min():.0f}-{R[:, 3].max():.0f} deg/step, median {np.median(R[:, 3]):.0f}; models turning at least 20 deg/step: {np.sum(R[:, 3] >= 20)} of {len(R)}")

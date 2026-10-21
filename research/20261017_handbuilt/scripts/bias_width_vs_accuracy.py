"""Does the typical size of a model's close-call biases (ignoring sign) predict its accuracy?
For each model: (1) second-third tie bias for all 720 placements of the top three (largest 1.6,
others -0.46; bias = third's value at which it starts to win, minus 1), and the same for the
largest-second tie (second's value at which the model switches to the largest's slot, minus the
largest's value 1.6, scaled by 1.6); (2) accuracy on 4M random lists. Models: ARC, zoo seeds 0-4,
run 3's 16 new seeds, 32 copies of ARC at 0.5% weight noise and 16 at 1%."""
import itertools
import json
from pathlib import Path

import numpy as np
import torch as th

from algzoo.core import accuracy, example, forward, load, zoo

pl = np.array(list(itertools.permutations(range(10), 3)))
N = len(pl); r = np.arange(N)
dev = "cuda"


def flip_point(P, which):
    base = np.full((N, 10), -0.46); base[r, pl[:, 0]] = 1.6
    if which == "23":
        base[r, pl[:, 1]] = 1.0; lo, hi = np.full(N, 0.8), np.full(N, 1.2); var, a, b = 2, 1, 2  # vary third; compare slots p2, p3
    else:
        base[r, pl[:, 2]] = 0.5; lo, hi = np.full(N, 1.3), np.full(N, 1.9); var, a, b = 1, 1, 0  # vary second near 1.6; compare p2 (right) vs p1
    for _ in range(26):
        mid = (lo + hi) / 2
        x = base.copy(); x[r, pl[:, var]] = mid
        s = forward(P, th.tensor(x, device=dev)).cpu().numpy()
        if which == "23":
            flip = s[r, pl[:, b]] > s[r, pl[:, a]]   # third's slot beats second's
        else:
            flip = s[r, pl[:, b]] > s[r, pl[:, a]]   # largest's slot beats second's: wrong side
        hi = np.where(flip, mid, hi); lo = np.where(flip, lo, mid)
    m = (lo + hi) / 2
    return m - 1.0 if which == "23" else (m - 1.6) / 1.6


models = {"ARC": ("trained", example())}
for s in range(5):
    models[f"zoo{s}"] = ("trained", zoo(16, 10, s))
for f in sorted(Path("/data/output/research/algzoo/20261016_run3/models").glob("2nd_argmax_16_10_*_1*.pth")):
    models["new" + f.stem.split("_")[-1]] = ("trained", load(str(f)))
rng = np.random.default_rng(123)
for sd, n in ((0.005, 32), (0.01, 16)):
    for i in range(n):
        models[f"noise{sd}_{i}"] = (f"noise {sd:.1%}", tuple(w * (1 + sd * rng.standard_normal(w.shape)) for w in example()))
rows = {}
for name, (kind, P) in models.items():
    b23, b12 = flip_point(P, "23"), flip_point(P, "12")
    acc, _ = accuracy(P, n_samples=4 * 2**20, dtype=th.float32, seed=1)
    rows[name] = dict(kind=kind, acc=acc, w23=float(np.mean(np.abs(b23))), w12=float(np.mean(np.abs(b12))),
                      m23=float(np.mean(b23)), m12=float(np.mean(b12)))
json.dump(rows, open("bias_width_rows.json", "w"), indent=1)
for kind in ("trained", "noise 0.5%", "noise 1.0%"):
    ks = [k for k in rows if rows[k]["kind"] == kind]
    err = np.array([1 - rows[k]["acc"] for k in ks])
    for w in ("w23", "w12"):
        v = np.array([rows[k][w] for k in ks])
        print(f"{kind:11s} n={len(ks):2d}  {w}: mean {v.mean():.4f}  corr with error rate {np.corrcoef(v, err)[0, 1]:+.2f}  (rank corr {np.corrcoef(np.argsort(np.argsort(v)), np.argsort(np.argsort(err)))[0, 1]:+.2f})")
    print(f"            error rate range {err.min():.3f}-{err.max():.3f}")

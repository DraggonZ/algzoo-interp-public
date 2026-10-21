"""Error rate of each trained network on lists with at most one positive number (the estimate assumes 8/9)."""
from pathlib import Path

import numpy as np

from algzoo.core import example, load, zoo

models = {"ARC": example()}
for s in range(5):
    models[f"zoo{s}"] = zoo(16, 10, s)
for f in sorted(Path("/data/output/research/algzoo/20261016_run3/models").glob("2nd_argmax_16_10_*_1*.pth")):
    models["new" + f.stem.split("_")[-1]] = load(str(f))
rng = np.random.default_rng(0)
X = rng.standard_normal((400_000, 10)); X = X[(X > 0).sum(1) <= 1]
y = np.argsort(-X, 1)[:, 1]
errs = {}
for k, (wi, wh, wo) in models.items():
    h = np.zeros((len(X), 16))
    for t in range(10):
        h = np.maximum(0, h @ wh.T + X[:, t:t + 1] * wi)
    errs[k] = np.mean((h @ wo.T).argmax(1) != y)
print(f"{len(X)} lists; error per network:", " ".join(f"{k} {v:.2f}" for k, v in errs.items()))
print(f"range {min(errs.values()):.2f}-{max(errs.values()):.2f}; median {np.median(list(errs.values())):.2f}; formula assumes {8/9:.2f}")

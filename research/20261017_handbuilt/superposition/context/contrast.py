"""Is the near-tie wall trivial (every solver errs near ties) or special to how training solved the task? Same test as
learnable2.py on hand-built networks: how big a tree on positions and the two top-three ratios is needed to predict
each network's near-tie decision (names the third or not)."""
import sys; sys.path.insert(0, "../steer")
import numpy as np, torch
from sklearn.ensemble import HistGradientBoostingClassifier
from harness import *
from algzoo.handbuilt_v19 import build_v19
from algzoo.handbuilt_v21 import build_v21
X = lists(4_000_000, seed=101); o = X.argsort(1, descending=True); v = X.sort(1, descending=True).values
near = (v[:, 1] > 0) & ((v[:, 1] - v[:, 2]) < 0.05 * v[:, 1]); X, o, v = X[near][:400000], o[near][:400000], v[near][:400000]
F = np.c_[o[:, :3].cpu().numpy().astype(float), (v[:, 2] / v[:, 1]).cpu().numpy(), (v[:, 0] / v[:, 1]).cpu().numpy()]
for name, W in (("ARC (trained)", ARC), ("hand-built v19 (16 neurons)", T(build_v19(kappa=0.5).params)), ("hand-built v21 (28 neurons)", T(build_v21()))):
    y = (scores(W, X).argmax(1) == o[:, 2]).cpu().numpy(); base = max(y.mean(), 1 - y.mean()); row = []
    for leaves, it in ((2, 1), (8, 10), (8, 50), (16, 100), (63, 400)):
        clf = HistGradientBoostingClassifier(max_iter=it, learning_rate=0.1 if it > 1 else 1.0, max_leaf_nodes=leaves).fit(F[:300000], y[:300000])
        row.append(f"{it*leaves}: {100*(clf.predict(F[300000:]) == y[300000:]).mean():.1f}")
    print(f"{name:28s} names the third on {100*y.mean():.1f}% of near ties (always-guess {100*base:.1f}%) | held-out accuracy by tree size (numbers: %): " + " | ".join(row), flush=True)

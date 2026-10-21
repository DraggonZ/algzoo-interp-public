"""Is the vote spread over many neurons? For lists where the second-largest is above 0.3, nudge it by 0.01 and split
the change in its position's final score over neurons (w_o[pos, n] x dh_n). Effective number of neurons carrying the
vote: participation ratio (sum |c|)^2 / sum c^2, median over lists. Also the share of the total |push| that cancels
(1 - |sum c| / sum |c|). Trained networks and the twin; rank correlation with loss at 1% noise (robust_across.log)."""
import re
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from algzoo.core import example, zoo, load
from algzoo.handbuilt_v18 import build_v18
import simlib

loss = {m.group(1): float(m.group(2)) for m in re.finditer(r"^(\S+)\s+acc .*loss at 0.01 ([0-9.]+)", open("robust_across.log").read(), re.M)}
models = {"ARC": example()}
for s in range(5):
    models[f"zoo{s}"] = zoo(16, 10, s)
for f in sorted(Path("/data/output/research/algzoo/20261016_run3/models").glob("2nd_argmax_16_10_*_1*.pth")):
    models["new" + f.stem.split("_")[-1]] = load(str(f))
models["twin"] = build_v18().params
X = simlib.lists(1818, 30_000); y = X.argsort(1)[:, -2]; r = np.arange(len(X)); sel = X[r, y] > 0.3
X2 = X.copy(); X2[r, y] += 0.01; rows = []
for name, w in models.items():
    P = tuple(np.asarray(p, float) for p in w); P = (P[0].reshape(-1), P[1], P[2])
    h1 = simlib.run(X, params=P)[2][:, 9]; h2 = simlib.run(X2, params=P)[2][:, 9]
    c = (P[2][y] * (h2 - h1) / 0.01)[sel]
    pr = np.median(np.abs(c).sum(1) ** 2 / np.maximum((c ** 2).sum(1), 1e-12)); canc = np.median(1 - np.abs(c.sum(1)) / np.maximum(np.abs(c).sum(1), 1e-12))
    rows.append((name, pr, canc)); print(f"{name:7s}: neurons carrying the vote (participation) {pr:.1f}; share of pushes that cancel {canc:.2f}; loss at 1% noise {loss.get(name, float('nan')):.3f}", flush=True)
tr = [(p, c, loss[n]) for n, p, c in rows if n in loss]
print(f"rank correlation with loss at 1% noise: participation {spearmanr([t[0] for t in tr], [t[2] for t in tr])[0]:.2f}; cancelling share {spearmanr([t[1] for t in tr], [t[2] for t in tr])[0]:.2f}")

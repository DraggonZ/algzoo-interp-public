"""Why do accurate networks have wider relative margins: bigger leads or smaller pushes? Per trained network: median
lead of the winner over the runner-up, median sum of absolute pushes on that lead, the second-largest's vote per unit
(slope of its position's score when its value is nudged by 0.01, lists where the second is above 0.3), and the typical
neuron value. Rank correlations with accuracy. 50,000 lists."""
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from algzoo.core import example, zoo, load
import simlib

models = {"ARC": example()}
for s in range(5):
    models[f"zoo{s}"] = zoo(16, 10, s)
for f in sorted(Path("/data/output/research/algzoo/20261016_run3/models").glob("2nd_argmax_16_10_*_1*.pth")):
    models["new" + f.stem.split("_")[-1]] = load(str(f))
X = simlib.lists(1616, 50_000); y = X.argsort(1)[:, -2]; B = len(X); r = np.arange(B)
sel = X[r, y] > 0.3; X2 = X.copy(); X2[r, y] += 0.01
rows = []
for name, w in models.items():
    P = tuple(np.asarray(p, float) for p in w); P = (P[0].reshape(-1), P[1], P[2])
    L, _, H = simlib.run(X, params=P); k = L.argmax(1); j = np.argsort(L, 1)[:, -2]
    lead = np.median(L[r, k] - L[r, j]); push = np.median(np.abs((P[2][k] - P[2][j]) * H[:, 9]).sum(1))
    vote = np.median(((simlib.run(X2, params=P)[0][r, y] - L[r, y]) / 0.01)[sel])
    typ = np.median([np.median(H[:, :, n][H[:, :, n] > 0]) for n in range(16) if (H[:, :, n] > 0).any()])
    acc = (k == y).mean(); rows.append((acc, lead, push, vote, typ))
    print(f"{name:7s} acc {acc:.3f}: median lead {lead:6.2f}, pushes on it {push:7.1f}, second's vote per unit {vote:6.1f}, typical neuron value {typ:5.1f}", flush=True)
R = np.array(rows)
for i, lab in ((1, "lead"), (2, "pushes"), (3, "vote per unit"), (4, "typical value")):
    rho, p = spearmanr(R[:, 0], R[:, i]); print(f"accuracy vs {lab}: rank correlation {rho:.2f} (p {p:.3f})")
rho, p = spearmanr(R[:, 0], R[:, 3] / R[:, 2]); print(f"accuracy vs vote per unit / pushes: {rho:.2f} (p {p:.3f})")

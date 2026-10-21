"""Hypothesis (from twin_push_parts.log): switching is the price of small levels, and small levels mean less cancellation.
Across 22 trained networks: switching rate (share of neuron-steps, at steps 2-9 that are not a new largest, where a
neuron's on/off state differs from the previous step) and the share of neuron-steps near threshold (0 < h < 5% of the
neuron's typical active value), against lead/pushes (robust_across.log) and accuracy. 20,000 lists."""
import re
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from algzoo.core import example, zoo, load
import simlib

mp = {m.group(1): (float(m.group(2)), float(m.group(3))) for m in re.finditer(r"^(\S+)\s+acc ([0-9.]+) cutoff [0-9.]+ margin/pushes ([0-9.]+)", open("robust_across.log").read(), re.M)}
models = {"ARC": example()}
for s in range(5):
    models[f"zoo{s}"] = zoo(16, 10, s)
for f in sorted(Path("/data/output/research/algzoo/20261016_run3/models").glob("2nd_argmax_16_10_*_1*.pth")):
    models["new" + f.stem.split("_")[-1]] = load(str(f))
X = simlib.lists(55, 20_000); V = simlib.task_vars(X); quiet = (V["newmax"] == 0); quiet[:, :2] = False
rows = []
for name, w in models.items():
    P = tuple(np.asarray(p, float) for p in w); P = (P[0].reshape(-1), P[1], P[2]); _, _, H = simlib.run(X, params=P)
    on = H > 0; flips = (on[:, 1:] != on[:, :-1])[quiet[:, 1:]].mean()
    typ = np.array([np.median(H[:, :, n][H[:, :, n] > 0]) if (H[:, :, n] > 0).any() else 1 for n in range(16)])
    near = ((H > 0) & (H < 0.05 * typ)).mean()
    acc, lp = mp[name]; rows.append((lp, acc, flips, near))
    print(f"{name:7s} lead/pushes {lp:.3f} acc {acc:.3f}: switching rate {flips:.3f}; near threshold {near:.3f}", flush=True)
R = np.array(rows)
for j, lab in ((2, "switching rate"), (3, "near threshold")):
    print(f"rank correlation of {lab} with lead/pushes {spearmanr(R[:, 0], R[:, j])[0]:.2f} (p {spearmanr(R[:, 0], R[:, j])[1]:.3f}); with accuracy {spearmanr(R[:, 1], R[:, j])[0]:.2f} (p {spearmanr(R[:, 1], R[:, j])[1]:.3f})")

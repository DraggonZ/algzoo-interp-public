"""Are close-call biases larger in models whose hidden timer is made of larger cancelling parts?
For each trained model: the 'wipe group' = neurons with strongly negative input weight (< -3).
Cancellation ratio = typical push of that group on the scores for a lone number (ages 2-8, background 0)
divided by the typical winning margin on two-number lists. Correlate with the one-sided bias size."""
import json
from pathlib import Path

import numpy as np

from algzoo.core import example, load, zoo

v2 = json.load(open("bias_formula_v2.json"))
models = {"ARC": example(), **{f"zoo{s}": zoo(16, 10, s) for s in range(5)}}
for f in sorted(Path("/data/output/research/algzoo/20261016_run3/models").glob("2nd_argmax_16_10_*_1*.pth")):
    models["new" + f.stem.split("_")[-1]] = load(str(f))


def run(P, x):
    wi, wh, wo = P; h = np.zeros(len(wi))
    for v in x: h = np.maximum(0, wh @ h + wi * v)
    return h


rows = []
for name, P in models.items():
    wi, wh, wo = P
    G = np.where(wi < -3)[0]
    if len(G) == 0:
        continue
    pushes, votes = [], []
    for p in range(1, 8):
        x = np.zeros(10); x[p] = 1.0; h = run(P, x)
        s = wo[:, G] @ h[G]; pushes.append(np.ptp(s))
    for p1 in range(10):
        for p2 in range(10):
            if p1 != p2:
                x = np.zeros(10); x[p1] = 1.5; x[p2] = 1.0; s = wo @ run(P, x); s -= s.mean()
                votes.append(np.sort(s)[-1] - np.sort(s)[-2])
    ratio = np.median(pushes) / np.median(votes)
    rows.append((name, len(G), ratio, v2[name]["a"], v2[name]["err"]))
r = np.array([(x[2], x[3], x[4]) for x in rows])
for x in rows:
    print(f"{x[0]:7s} wipe group {x[1]} neurons, cancellation ratio {x[2]:5.1f}, bias term {x[3]:.4f}, error {100*x[4]:.1f}")
print(f"correlation of cancellation ratio with bias size {np.corrcoef(r[:,0], r[:,1])[0,1]:+.2f}, with error {np.corrcoef(r[:,0], r[:,2])[0,1]:+.2f} (n={len(r)})")

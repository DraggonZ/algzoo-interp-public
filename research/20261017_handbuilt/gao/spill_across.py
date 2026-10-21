"""Does each network erase the neighbour its revealed wave spills onto? For each network: lists with a record P at
position 3 (P from 0.8 to 1.2), negative numbers elsewhere except a bigger number (1.6 P) at position 7, so the record is
overtaken. The revealed vote's profile: d score[3 + k] / d P for k = 1, 2, relative to k = 0 (median over lists).
Compared with the erasure ratios in erasure_across.log."""
from pathlib import Path
import re
import numpy as np
from algzoo.core import example, zoo, load
import simlib
models = {"ARC": example()}
for s in range(5):
    models[f"zoo{s}"] = zoo(16, 10, s)
for f in sorted(Path("/data/output/research/algzoo/20261016_run3/models").glob("2nd_argmax_16_10_*_1*.pth")):
    models["new" + f.stem.split("_")[-1]] = load(str(f))
rng = np.random.default_rng(5); B = 20_000
X = -np.abs(rng.standard_normal((B, 10))) * 0.5 - 0.2; Pv = rng.uniform(0.8, 1.2, B); X[:, 3] = Pv; X[:, 7] = 1.6 * Pv
er = {}
for l in open("erasure_across.log"):
    m = re.match(r"^(\S+)\s+([-0-9.]+)/\s*([-0-9.]+)\s+([-0-9.]+)/\s*([-0-9.]+)\s+([-0-9.]+)/\s*([-0-9.]+)\s+([-0-9.]+)/\s*([-0-9.]+)", l)
    if m:
        v = list(map(float, m.groups()[1:])); er[m.group(1)] = ((v[1] / v[0] + v[3] / v[2]) / 2, (v[5] / v[4] + v[7] / v[6]) / 2)
print(f"{'network':8s} {'spill r+1':>9s} {'spill r+2':>9s} {'erasure r+1':>11s} {'erasure r+2':>11s}")
rows = []
for name, w in models.items():
    P = tuple(np.asarray(p, float) for p in w); P = (P[0].reshape(-1), P[1], P[2])
    X2 = X.copy(); X2[:, 3] += 0.01
    d = (simlib.run(X2, params=P)[0] - simlib.run(X, params=P)[0]) / 0.01
    s1, s2 = np.median(d[:, 4] / d[:, 3]), np.median(d[:, 5] / d[:, 3]); rows.append((s1, s2) + er[name])
    print(f"{name:8s} {s1:9.2f} {s2:9.2f} {er[name][0]:11.2f} {er[name][1]:11.2f}", flush=True)
r = np.array(rows)
print("correlation across networks: spill r+1 vs erasure r+1 %.2f; spill r+2 vs erasure r+2 %.2f" % (np.corrcoef(r[:, 0], r[:, 2])[0, 1], np.corrcoef(r[:, 1], r[:, 3])[0, 1]))

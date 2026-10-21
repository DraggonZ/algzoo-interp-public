"""Per-neuron points for the figure: variance matched and accuracy with that neuron simulated alone, for the 22 trained
networks and both hand-built ones, using the role-free formulas of gao/sim_across.py. Saves sim_across_points.json."""
import json
from pathlib import Path
import numpy as np
from algzoo.core import example, zoo, load
from algzoo.handbuilt import build_v5
from algzoo.handbuilt_v18 import build_v18
import simlib
from simlib import lists
from sim_across import feats
models = {"ARC": example()}
for s in range(5):
    models[f"zoo{s}"] = zoo(16, 10, s)
for f in sorted(Path("/data/output/research/algzoo/20261016_run3/models").glob("2nd_argmax_16_10_*_1*.pth")):
    models["new" + f.stem.split("_")[-1]] = load(str(f))
models["v5"] = build_v5().params; models["v18"] = build_v18().params
Xf, Xs = lists(101, 50_000), lists(202, 50_000); Ff, Fs = feats(Xf), feats(Xs); A = Ff.reshape(-1, Ff.shape[-1])
y = Xs.argsort(1)[:, -2]; out = {}
for name, w in models.items():
    P = tuple(np.asarray(p, float) for p in w); P = (P[0].reshape(-1), P[1], P[2])
    _, Pf, _ = simlib.run(Xf, params=P); Ls, _, Hs = simlib.run(Xs, params=P); base = float((Ls.argmax(1) == y).mean()); pts = []
    for n in range(16):
        v = Hs[:, :, n].var()
        if v < 1e-9:
            continue
        co = np.linalg.lstsq(A, Pf[:, :, n].reshape(-1), rcond=None)[0]; sim = np.maximum(Fs @ co, 0)
        r2 = 1 - ((sim - Hs[:, :, n]) ** 2).mean() / v
        pts.append([n, float(r2), float((simlib.run(Xs, {n: sim}, params=P)[0].argmax(1) == y).mean())])
    out[name] = {"base": base, "points": pts}; print(name, base, len(pts), flush=True)
json.dump(out, open("sim_across_points.json", "w"))

"""Is the erasure of the vote right after an overtaken record shared by other trained networks? For each of the 22
trained networks (and the twin): own-score response per unit of a kept non-record number y at position q = r + 1 right
after a record at r (q = 4 and 5), median over random lists, split by whether the record is later overtaken. Ratio
overtaken / stays near 0 means erased. Same for y two places after (r + 2) as a reference. 400,000 lists, seed 33."""
from pathlib import Path
import numpy as np
from algzoo.core import example, zoo, load
from algzoo.handbuilt_v18 import build_v18
import simlib
models = {"ARC": example()}
for s in range(5):
    models[f"zoo{s}"] = zoo(16, 10, s)
for f in sorted(Path("/data/output/research/algzoo/20261016_run3/models").glob("2nd_argmax_16_10_*_1*.pth")):
    models["new" + f.stem.split("_")[-1]] = load(str(f))
models["twin"] = build_v18().params
X = np.random.default_rng(33).standard_normal((400_000, 10)); B = len(X); eps = 0.01
m1p = np.stack([np.max(np.c_[np.zeros((B, 2)), X[:, :t]], 1) for t in range(10)], 1); rec = X > m1p
print(f"{'network':8s} " + "  ".join(f"r+{k} q={q}: stays/overtaken" for k in (1, 2) for q in (4, 5)))
ratios = {}
for name, w in models.items():
    P = tuple(np.asarray(p, float) for p in w); P = (P[0].reshape(-1), P[1], P[2]); line = f"{name:8s} "; rr = []
    for k in (1, 2):
        for q in (4, 5):
            r = q - k
            base = rec[:, r] & ~rec[:, r + 1:q + 1].any(1) & (X[:, q] > 0.5 * X[:, r]) & (X[:, q] < X[:, r] - 2 * eps)
            over = base & rec[:, q + 1:].any(1); stay = base & ~rec[:, q + 1:].any(1); out = []
            for mm in (stay, over):
                Z = X[mm]; Z2 = Z.copy(); Z2[:, q] += eps
                out.append(np.median((simlib.run(Z2, params=P)[0][:, q] - simlib.run(Z, params=P)[0][:, q]) / eps))
            line += f"  {out[0]:6.2f}/{out[1]:6.2f}"; rr.append(out[1] / out[0])
    ratios[name] = rr; print(line, flush=True)
tr = np.array([v for k, v in ratios.items() if k != "twin"])
print("ratio overtaken/stays, median over trained networks: r+1 q=4 %.2f, q=5 %.2f; r+2 q=4 %.2f, q=5 %.2f" % tuple(np.median(tr, 0)))
print("networks with r+1 ratio below 0.3 at both q:", [k for k, v in ratios.items() if v[0] < 0.3 and v[1] < 0.3])

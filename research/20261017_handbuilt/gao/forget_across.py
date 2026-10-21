"""Do all trained networks forget small numbers, and at what cutoff? For ARC's and 21 other trained networks (as in
sim_across.py), nudge one positive number at position 3 or 5 by +0.01 and measure how far the final state (all 16
neurons) moves per unit. Bin the number by its ratio to the second-largest before it (excluding the list's top two and
records). Report each bin's median movement relative to numbers at 0.9-1.0 of the second-largest; a network that forgets
below a cutoff shows near-zero values below it. 100,000 lists."""
from pathlib import Path
import numpy as np
from algzoo.core import example, zoo, load
import simlib

models = {"ARC": example()}
for s in range(5):
    models[f"zoo{s}"] = zoo(16, 10, s)
for f in sorted(Path("/data/output/research/algzoo/20261016_run3/models").glob("2nd_argmax_16_10_*_1*.pth")):
    models["new" + f.stem.split("_")[-1]] = load(str(f))
X = simlib.lists(505, 100_000); B = len(X)
edges = [0, 0.25, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
print("bins of number / second-largest before it: " + ", ".join(f"{a}-{b}" for a, b in zip(edges[:-1], edges[1:])))
for name, w in models.items():
    P = tuple(np.asarray(p, float) for p in w); P = (P[0].reshape(-1), P[1], P[2])
    acc = (simlib.run(X, params=P)[0].argmax(1) == X.argsort(1)[:, -2]).mean()
    H = simlib.run(X, params=P)[2][:, 9]
    r, gs = [], []
    for s in (3, 5):
        Pr = np.c_[np.zeros((B, 2)), X[:, :s]]; m2 = np.sort(Pr, 1)[:, -2]; m1 = Pr.max(1)
        top2 = np.argsort(X, 1)[:, -2:]; ok = ~(top2 == s).any(1) & (X[:, s] > 0) & (X[:, s] < m1) & (m2 > 0)
        X2 = X.copy(); X2[:, s] += 0.01
        g = np.linalg.norm(simlib.run(X2, params=P)[2][:, 9] - H, axis=1) / 0.01
        r.append(np.where(ok, X[:, s] / np.where(m2 > 0, m2, 1), np.nan)); gs.append(g)
    r, g = np.concatenate(r), np.concatenate(gs)
    med = [np.median(g[(r >= a) & (r < b)]) for a, b in zip(edges[:-1], edges[1:])]
    print(f"{name:7s} acc {acc:.3f}: " + " ".join(f"{m / med[-1]:5.2f}" for m in med), flush=True)

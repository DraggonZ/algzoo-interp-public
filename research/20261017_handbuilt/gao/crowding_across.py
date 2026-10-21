"""Why do networks that never forget do worse? Crowding hypothesis: each kept small number adds an imperfect vote, so
errors grow with the number of positive numbers in a list, faster for networks that keep them all. Per network:
error rate by count of positive numbers (bins 2-3, 4-5, 6-7, 8-10), and by kind of mistake (names the third-largest
within 10%, the third beyond 10%, the largest, fourth or lower). Forgetting cutoffs from forget_across.log. 200,000 lists."""
import re
from pathlib import Path
import numpy as np
from algzoo.core import example, zoo, load
import simlib

cut = {m.group(1): float(m.group(2)) for m in re.finditer(r"  (\S+)\s+cutoff ([0-9.]+)", open("forget_across.log").read())}
models = {"ARC": example()}
for s in range(5):
    models[f"zoo{s}"] = zoo(16, 10, s)
for f in sorted(Path("/data/output/research/algzoo/20261016_run3/models").glob("2nd_argmax_16_10_*_1*.pth")):
    models["new" + f.stem.split("_")[-1]] = load(str(f))
X = simlib.lists(1001, 200_000); o = np.argsort(X, 1); y = o[:, -2]; npos = (X > 0).sum(1)
r32 = X[np.arange(len(X)), o[:, -3]] / X[np.arange(len(X)), y]
bins = [(2, 3), (4, 5), (6, 7), (8, 10)]
print("network cutoff acc | error % by positives " + " ".join(f"{a}-{b}" for a, b in bins) + " | mistakes naming: 3rd within 10%, 3rd beyond, largest, 4th or lower (% of lists)")
rows = []
for name, w in models.items():
    P = tuple(np.asarray(p, float) for p in w); P = (P[0].reshape(-1), P[1], P[2])
    a = simlib.run(X, params=P)[0].argmax(1); wrong = a != y
    eb = [100 * wrong[(npos >= lo) & (npos <= hi)].mean() for lo, hi in bins]
    rank = np.argsort(o, 1)[np.arange(len(X)), a]  # 9 = largest, 8 = second, 7 = third ...
    k = [100 * np.mean(wrong & (rank == 7) & (r32 > 0.9)), 100 * np.mean(wrong & (rank == 7) & (r32 <= 0.9)),
         100 * np.mean(wrong & (rank == 9)), 100 * np.mean(wrong & (rank <= 6))]
    rows.append((name, cut.get(name, np.nan), 1 - wrong.mean(), eb, k))
    print(f"{name:7s} {cut.get(name, np.nan):.2f} {1 - wrong.mean():.3f} | " + " ".join(f"{e:5.1f}" for e in eb) + " | " + " ".join(f"{v:4.1f}" for v in k), flush=True)
c = np.array([r[1] for r in rows])
for lab, sel in (("never forget", c == 0), ("cutoff 0.25-0.5", (c > 0) & (c < 0.6)), ("cutoff >= 0.6", c >= 0.6)):
    E = np.array([r[3] for r in rows])[sel].mean(0); K = np.array([r[4] for r in rows])[sel].mean(0)
    print(f"mean, {lab:15s} ({sel.sum():2d}): error by positives " + " ".join(f"{e:5.1f}" for e in E) + " | kinds " + " ".join(f"{v:4.1f}" for v in K))

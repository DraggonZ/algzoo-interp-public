"""Do the cancelling neurons of other trained networks also depend on kept numbers? In each of the 22 trained networks,
take as cancelling neurons those silenced by a new largest (zero at >= 95% of steps where the input beats the largest so
far, and on at >= 50% of other steps after step 1). Resample them together under (a) their role claim (input, largest
and second-largest with places, previous record, the number after the current largest) and (b) the memory claim
(every kept number at its place, plus the current and previous input), and report accuracy. Twin (v18): its clock and
flip neurons. 30,000 lists, seed 202."""
from pathlib import Path
import numpy as np
from algzoo.core import example, zoo, load
from algzoo.handbuilt_v18 import build_v18
import scrub
import simlib
from simlib import lists, task_vars
models = {"ARC": example()}
for s in range(5):
    models[f"zoo{s}"] = zoo(16, 10, s)
for f in sorted(Path("/data/output/research/algzoo/20261016_run3/models").glob("2nd_argmax_16_10_*_1*.pth")):
    models["new" + f.stem.split("_")[-1]] = load(str(f))
models["twin"] = build_v18().params
import sys
THR = float(sys.argv[1]) if len(sys.argv) > 1 else 0.95
X = lists(202, 30_000); y = X.argsort(1)[:, -2]; V = task_vars(X)
rec = (V["newmax"] > 0) & (np.arange(10)[None, :] >= 1); other = (V["newmax"] == 0) & (np.arange(10)[None, :] >= 2)
print(f"{'network':8s} {'acc':>6s} {'cancelling neurons':24s} {'role claim':>10s} {'kept-numbers claim':>18s}")
for name, w in models.items():
    P = tuple(np.asarray(p, float) for p in w); P = (P[0].reshape(-1), P[1], P[2]); scrub.PARAMS = P
    L, _, H = simlib.run(X, params=P); base = (L.argmax(1) == y).mean()
    canc = [n for n in range(16) if (H[:, :, n][rec] <= 1e-9).mean() >= THR and (H[:, :, n][other] > 1e-9).mean() >= 0.5]
    if not canc:
        print(f"{name:8s} {base:6.3f} none found"); continue
    scrub.GROUPS["cancellers"] = canc
    res = []
    for opts in (("prevrec", "after"), ("asmemory", "cur", "prev")):
        sub = scrub.scrubbed_values(X, ["cancellers"], opts={"cancellers": opts})
        res.append((simlib.run(X, sub, params=P)[0].argmax(1) == y).mean())
    print(f"{name:8s} {base:6.3f} {str(canc):24s} {res[0]:10.3f} {res[1]:18.3f}", flush=True)

"""Does the store of the largest carry timing in other trained networks too? In each of the 22 trained networks (and
the twin), take the neuron whose activity best tracks the largest so far (highest correlation with it at steps 5-9),
resample only that neuron under the claim 'depends on the input and the largest's value only' (largest moved to a random
earlier place; gao/scrub.py's store claim plus the second-largest kept), and under the same claim with the largest's place
kept. If the store carries timing, the first loses accuracy and the second does not. 50,000 lists, seed 202."""
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
X = lists(202, 50_000); y = X.argsort(1)[:, -2]; V = task_vars(X)
print(f"{'network':8s} {'store':>5s} {'corr':>5s} {'acc':>6s} {'largest moved':>13s} {'place kept':>10s}")
rows = []
for name, w in models.items():
    P = tuple(np.asarray(p, float) for p in w); P = (P[0].reshape(-1), P[1], P[2]); scrub.PARAMS = P
    L, _, H = simlib.run(X, params=P); base = (L.argmax(1) == y).mean()
    corr = [np.corrcoef(H[:, 5:, n].ravel(), V["m1in"][:, 5:].ravel())[0, 1] if H[:, 5:, n].std() > 0 else 0 for n in range(16)]
    n = int(np.nanargmax(corr)); scrub.GROUPS["store_x"] = [n]
    res = []
    for opts in (("m2",), ("m1pos", "m2")):
        sub = scrub.scrubbed_values(X, ["store_x"], opts={"store_x": opts}) if False else None
        V2 = np.zeros((len(X), 10))
        for t in range(10):
            X2 = scrub.claim_list(X, t, "store+detector", opts); V2[:, t] = scrub.state_at(X2, t)[:, n]
        res.append((simlib.run(X, {n: V2}, params=P)[0].argmax(1) == y).mean())
    rows.append((name, base, res[0], res[1]))
    print(f"{name:8s} {n:5d} {corr[n]:5.2f} {base:6.3f} {res[0]:13.3f} {res[1]:10.3f}", flush=True)
r = np.array([[b, a, c] for _, b, a, c in rows if _ != "twin"])
print(f"trained networks: median loss with the largest moved {np.median(r[:, 0] - r[:, 1]) * 100:.1f} points, with its place kept {np.median(r[:, 0] - r[:, 2]) * 100:.1f}; "
      f"networks losing more than 2 points when moved: {np.sum(r[:, 0] - r[:, 1] > 0.02)}/{len(r)}")

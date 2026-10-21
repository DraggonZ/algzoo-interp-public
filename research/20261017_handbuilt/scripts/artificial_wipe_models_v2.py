"""v2: ten random control sets per network and written pass rules.
Sufficiency passes if the wiped set gives >= 0.6 and the controls' mean <= 0.2; necessity passes if the un-wiped
network is right in <= 0.3 and the controls' mean >= 0.6. Otherwise as artificial_wipe.py, for each of the 22 trained networks, using each network's wiped set (neurons exactly zero in
>= 95% of steps where a new largest arrives on random lists; see wipe_across_models.py). Two-number lists use
1.0 then 1.5; the right answer is the 1.0's position."""
import itertools
from pathlib import Path

import numpy as np

from algzoo.core import example, load, zoo

models = {"ARC": example()}
for s in range(5):
    models[f"zoo{s}"] = zoo(16, 10, s)
for f in sorted(Path("/data/output/research/algzoo/20261016_run3/models").glob("2nd_argmax_16_10_*_1*.pth")):
    models["new" + f.stem.split("_")[-1]] = load(str(f))
rng = np.random.default_rng(0)


def run(P, X, t_set=None, G=(), V=None):
    wi, wh, wo = P
    h = np.zeros((len(X), 16)); hs = []
    for t in range(10):
        h = np.maximum(0, h @ wh.T + X[:, t:t + 1] * wi)
        if t_set is not None and len(G):
            m = t_set == t
            h[np.ix_(m, G)] = 0 if V is None else V[m, t][:, G]
        hs.append(h)
    return (h @ wo.T).argmax(1), np.stack(hs, 1)


X = rng.standard_normal((60000, 10))
rm = np.maximum.accumulate(X, 1); newmax = np.zeros_like(X, bool); newmax[:, 1:] = (X[:, 1:] > rm[:, :-1]) & (rm[:, :-1] > 0)
pairs = np.array([(a, t) for a, t in itertools.permutations(range(10), 2) if t > a]); n = len(pairs); r = np.arange(n)
lone = np.zeros((n, 10)); lone[r, pairs[:, 0]] = 1.0
two = lone.copy(); two[r, pairs[:, 1]] = 1.5
print("network  acc  wiped | lone number, wipe by hand: wiped set / random same size | two numbers, un-wipe: wiped set / random")
for name, P in models.items():
    _, H = run(P, X); acc = np.mean(_ == np.argsort(-X, 1)[:, 1])
    W = [k for k in range(16) if (H[newmax][:, k] == 0).mean() >= 0.95]
    if not W:
        print(f"{name:7s} {acc:.3f}  none"); continue
    rest = [k for k in range(16) if k not in W]
    ctrl = [list(rng.choice(rest, len(W), replace=False)) for _ in range(10)]
    _, Hn = run(P, lone)
    suff = lambda G: np.mean(run(P, lone, pairs[:, 1], G)[0] == pairs[:, 0])
    nec = lambda G: np.mean(run(P, two, pairs[:, 1], G, Hn)[0] == pairs[:, 0])
    s1, c1, s2, c2 = suff(W), np.mean([suff(G) for G in ctrl]), nec(W), np.mean([nec(G) for G in ctrl])
    ok1, ok2 = s1 >= 0.6 and c1 <= 0.2, s2 <= 0.3 and c2 >= 0.6
    verdict = {2: "both", 1: "one", 0: "neither"}[int(ok1) + int(ok2)]
    print(f"{name:7s} {acc:.3f} {len(W):2d}    | {s1:.2f} / {c1:.2f} | {s2:.2f} / {c2:.2f} | {verdict}")

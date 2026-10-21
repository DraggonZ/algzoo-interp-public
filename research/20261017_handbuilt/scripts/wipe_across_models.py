"""Do the other trained networks share ARC's hidden copy and wipe? For each of 22 trained 16-neuron networks:
(1) the wiped set: neurons that are exactly zero in >= 95% of steps where a new largest arrives (random lists);
(2) hidden lone number: score spread at the end for a single 1 among zeros, against the spread for a two-number
    list (1.0, then 1.5 later), averaged over positions;
(3) swap test on random lists (as arc_interchange_random.py): copy the non-wiped neurons, the wiped ones, or a
    random set the size of the non-wiped ones at the step the largest arrives."""
import itertools
from pathlib import Path

import numpy as np

from algzoo.core import example, load, zoo

models = {"ARC": example()}
for s in range(5):
    models[f"zoo{s}"] = zoo(16, 10, s)
for f in sorted(Path("/data/output/research/algzoo/20261016_run3/models").glob("2nd_argmax_16_10_*_1*.pth")):
    models["new" + f.stem.split("_")[-1]] = load(str(f))


def run(P, X, patch_t=None, G=None, HD=None):
    wi, wh, wo = P
    h = np.zeros((len(X), 16)); hs = []
    for t in range(X.shape[1]):
        h = np.maximum(0, h @ wh.T + X[:, t:t + 1] * wi)
        if patch_t is not None and len(G):
            m = patch_t == t
            h[np.ix_(m, G)] = HD[m, t][:, G]
        hs.append(h)
    return h @ wo.T, np.stack(hs, 1)


rng = np.random.default_rng(0)
n = 60_000
X = rng.standard_normal((n, 10))
order = np.argsort(-X, 1); p1, a = order[:, 0], order[:, 1]
keep = (a < p1) & (p1 >= 2); Xs, p1, a = X[keep], p1[keep], a[keep]
a2 = np.array([rng.choice([j for j in range(q) if j != b]) for q, b in zip(p1, a)])
D = Xs.copy(); i = np.arange(len(Xs)); D[i, a], D[i, a2] = Xs[i, a2], Xs[i, a]
runmax = np.maximum.accumulate(X, 1)
newmax = np.zeros_like(X, bool); newmax[:, 1:] = (X[:, 1:] > runmax[:, :-1]) & (runmax[:, :-1] > 0)
two = [(q, r) for q, r in itertools.permutations(range(10), 2) if q < r]
print("network  acc | wiped set | lone-number spread / two-number spread | swap: rest, wiped, random same size")
for name, P in models.items():
    s, H = run(P, X); acc = np.mean(s.argmax(1) == order[:, 1])
    zero_frac = (H[newmax] == 0).mean(0)
    W = [k for k in range(16) if zero_frac[k] >= 0.95]; Rst = [k for k in range(16) if k not in W]
    lone = np.zeros((10, 10)); lone[np.arange(10), np.arange(10)] = 1.0
    sl, _ = run(P, lone)
    tw = np.zeros((len(two), 10)); tw[np.arange(len(two)), [q for q, _ in two]] = 1.0; tw[np.arange(len(two)), [r for _, r in two]] = 1.5
    st, _ = run(P, tw)
    spread = lambda S: np.mean(S.max(1) - S.min(1))
    sb, _ = run(P, Xs); sd, HD = run(P, D)
    ok = (sb.argmax(1) == a) & (sd.argmax(1) == a2)
    res = []
    for G in (Rst, W, sorted(rng.choice(16, len(Rst), replace=False))):
        sp, _ = run(P, Xs[ok], p1[ok], G, HD[ok]); res.append(np.mean(sp.argmax(1) == a2[ok]))
    print(f"{name:7s} {acc:.3f} | {len(W):2d} {str(W):22s} | {spread(sl) / spread(st):.2f} | {res[0]:.2f} {res[1]:.2f} {res[2]:.2f}")

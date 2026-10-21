"""Methods study 3: is 'two groups' pushes are mirror images' evidence of a designed pair of copies?
In the hand-built network the designed pair is clockA vs clockB. For a lone number, compare that split
with random splits of the 16 neurons, and with the 'wipe test': does a later bigger number reveal a vote?"""
import numpy as np

from algzoo import handbuilt as hb

net = hb.build_v2(100.0, 100.0); wi, wh, wo = net.params; idx = net.idx
A = [idx[f"clockA{k}"] for k in range(3)]; B = [idx[f"clockB{k}"] for k in range(3)]


def run(x):
    h = np.zeros(16)
    for v in x: h = np.maximum(0, wh @ h + wi * v)
    return h


c = lambda s: s - s.mean()
x = np.zeros(10); x[2] = 1.0; h = run(x)
def mirror(G):
    R = [k for k in range(16) if k not in G]
    a, b = c(wo[:, G] @ h[G]), c(wo[:, R] @ h[R])
    return np.ptp(a), np.ptp(a + b), (np.corrcoef(a, b)[0, 1] if np.ptp(a) > 1e-9 and np.ptp(b) > 1e-9 else np.nan)
print("designed split (clockA vs rest): group push spread %.1f, sum %.2g, corr %.2f" % mirror(A))
rng = np.random.default_rng(0)
res = np.array([mirror(list(rng.choice(16, 3, replace=False))) for _ in range(500)])
ok = ~np.isnan(res[:, 2])
print(f"random 3-neuron splits: {ok.sum()} of 500 push at all; among those, sum spread max {res[ok,1].max():.2g}, correlation median {np.median(res[ok,2]):.2f}")
print("So mirror-image pushes follow from the total being invisible; they do not single out a designed pair.")

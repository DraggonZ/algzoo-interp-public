import numpy as np
from algzoo.core import example
WI, WH, WO = (np.asarray(p, float) for p in example())
d = np.load("/data/output/research/algzoo/20261017_handbuilt/hilton/topdown/grid_lists.npz"); X, w = d["X"], d["w"]
Y = X.argsort(1)[:, -2]
def grid_acc(scores_fn, chunk=200_000):
    tot = 0.0
    for i in range(0, len(X), chunk):
        S = scores_fn(X[i:i + chunk]); tot += (w[i:i + chunk] * (S.argmax(1) == Y[i:i + chunk])).sum()
    return tot
def intact(Xc):
    h = np.zeros((len(Xc), 16))
    for t in range(10): h = np.maximum(h @ WH.T + Xc[:, t:t + 1] * WI, 0)
    return h @ WO.T

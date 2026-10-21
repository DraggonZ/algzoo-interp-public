"""Why does even the trained-scale twin cancel so much (lead / pushes about 0.003, against 0.022 for the trained
network; robust_twin.log)? Pushes on the lead (w_o[winner] - w_o[runner-up]) * h at the end, by neuron group, for the
trained-scale twin with a 1% tie-breaker and for the trained network. Twin groups: store/detector 0-1, keep gate 2,
flip 3, clock 4-6, flag 7, memory 8-15."""
import numpy as np
from algzoo.handbuilt_v18 import build_v18
import simlib

X = simlib.lists(31, 50_000); r = np.arange(len(X))
nets = {"trained-scale twin": (tuple(np.asarray(p, float) for p in build_v18(g=10, a=10, w=10, lam=10.0, c=-9.0, eps=0.1).params),
                              {"store+detector": [0, 1], "keep gate": [2], "flip": [3], "clock": [4, 5, 6], "flag": [7], "memory": list(range(8, 16))}),
        "lam 1 twin": (tuple(np.asarray(p, float) for p in build_v18(g=10, a=10, w=10, lam=1.0, c=-9.0, eps=0.1).params),
                       {"store+detector": [0, 1], "keep gate": [2], "flip": [3], "clock": [4, 5, 6], "flag": [7], "memory": list(range(8, 16))}),
        "trained": ((simlib.WI, simlib.WH, simlib.WO), {"store+detector": [2, 4], "gates": [5, 9], "cancelling": [1, 6, 7, 8], "memory": [0, 3, 10, 11, 12, 13, 14, 15]})}
for name, (P, G) in nets.items():
    P = (P[0].reshape(-1), P[1], P[2]); L, _, H = simlib.run(X, params=P)
    k = L.argmax(1); o = np.argsort(L, 1); j = o[:, -2]; lead = L[r, k] - L[r, j]
    push = (P[2][k] - P[2][j]) * H[:, -1]; tot = np.abs(push).sum(1); m = tot > 1e-9; push, tot, lead = push[m], tot[m], lead[m]
    parts = "; ".join(f"{g} {np.median(np.abs(push[:, ns]).sum(1) / tot):.2f} (net {np.median(np.abs(push[:, ns].sum(1)) / tot):.2f})" for g, ns in G.items())
    print(f"{name:20s}: lead/pushes median {np.median(lead / tot):.4f}; share of |pushes| by group (and of net group push): {parts}")

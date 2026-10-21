"""Positive control for the forgetting tests: a twin variant that forgets. Its keep gate subtracts 0.75 x the largest so
far, so a number below three-quarters of the largest writes no vote (every number below three-quarters of the
second-largest is then forgotten too, since the second is at most the largest). Votes of kept numbers shrink by
0.75 x the largest, so accuracy drops; the variant serves only as a network known to forget. Runs the output-level
resampling test (gao/scrub.py's memory claim, all groups) and the score-level nudge test (gao/forget_internal.py) on
the trained network, the twin and the forgetting twin."""
import numpy as np
from algzoo.handbuilt_v18 import build_v18
import scrub
import simlib
from simlib import lists


def forgetting_twin():
    net = build_v18(); wi, wh, wo = (np.asarray(p, float) for p in net.params)
    S, D, G = 0, 1, 2; A0 = 0.25; s, a = 20.0, 1000.0
    wh = wh.copy(); wh[G, S] += -A0 * 0.75 / s; wh[G, D] += -A0 * 0.75 / a   # gate: minus 0.75 x the largest so far
    return wi, wh, wo


def v7_params():
    import sys
    sys.path.insert(0, "/data/output/research/algzoo/20261017_handbuilt/scripts")
    from handbuilt_v7 import build_v7
    return tuple(np.asarray(p, float) for p in build_v7().params)


X = lists(303, 100_000); y = X.argsort(1)[:, -2]
nets = {"trained": ((simlib.WI, simlib.WH, simlib.WO), [0, 3, 10, 11, 12, 13, 14, 15]),
        "twin": (tuple(np.asarray(p, float) for p in build_v18().params), list(range(8, 16))),
        "forgetting twin": (forgetting_twin(), list(range(8, 16))),
        "v7 (forgets by the rule)": (v7_params(), None)}
X2 = scrub.claim_list(X, 9, "memory", ("cur", "prev"))      # forgotten numbers two or more steps old redrawn
for name, (P, mem) in nets.items():
    L = simlib.run(X, params=P)[0]; L2 = simlib.run(X2, params=P)[0]
    out = f"{name:26s}: accuracy {(L.argmax(1) == y).mean():.3f}; with forgotten numbers redrawn {(L2.argmax(1) == y).mean():.3f} (answers unchanged {(L2.argmax(1) == L.argmax(1)).mean():.3f})"
    rat = []
    for s_ in (3, 5):
        B = len(X); Pp = np.c_[np.zeros((B, 2)), X[:, :s_]]; m2 = np.sort(Pp, 1)[:, -2]; x = X[:, s_]
        top = (np.argsort(X, 1)[:, -2:] == s_).any(1)
        forg = (x > 0) & (x < 0.75 * m2) & ~top; kept = (x > 0) & (x >= 0.75 * m2) & ~top & (x < Pp.max(1))
        Xn = X.copy(); Xn[:, s_] += 0.01
        gs = np.linalg.norm(simlib.run(Xn, params=P)[0] - L, axis=1) / 0.01
        rat.append(np.median(gs[forg]) / np.median(gs[kept]))
    print(out + f"; score nudge ratio forgotten/kept: position 3 {rat[0]:.3f}, position 5 {rat[1]:.3f}", flush=True)

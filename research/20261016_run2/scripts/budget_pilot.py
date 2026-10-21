"""Pilot for budgeted enumeration at length 4 (review 2, direction 5): how concentrated is the wrong mass?

Records, for every final linear region of M16,4, its total mass and its wrong mass. Reports the share of the wrong
mass held by the top x% of regions ranked by (a) total mass, (b) wrong mass (oracle ordering, the best possible).
"""
import numpy as np

import exact_nd as E
from algzoo.core import load

rec = []
orig_measure = E.cone_measure
state = {"in_region": False}


def run(params):
    """Copy of exact_accuracy's control flow with per-region recording."""
    w_hi, w_hh, w_oh = params
    n = w_oh.shape[0]
    d = w_hh.shape[0]
    ops = E.Ops()
    order_planes = [(i, j) for i in range(n) for j in range(i + 1, n)]

    def finish(R, H, A):
        L = w_oh @ A
        total = orig_measure(R, ops)
        pieces = [(R, H)]
        for i, j in order_planes:
            v = np.zeros(n)
            v[i], v[j] = 1.0, -1.0
            nxt = []
            for R2, H2 in pieces:
                pl, mi = E.cut(R2, H2, v, ops)
                nxt += [c for c in (pl, mi) if c is not None]
            pieces = nxt
        wrong = 0.0
        for R2, H2 in pieces:
            tgt = int(np.argsort(R2.mean(0))[-2])
            cur = (R2, H2)
            for k in range(n):
                if k == tgt or cur is None:
                    continue
                D = L[tgt] - L[k]
                if np.abs(D).max() < 1e-12 * (1 + np.abs(L).max()):
                    if k < tgt:
                        wrong += orig_measure(cur[0], ops)
                        cur = None
                    continue
                cur, bad = E.cut(cur[0], cur[1], D, ops)
                if bad is not None:
                    wrong += orig_measure(bad[0], ops)
        rec.append((total, wrong))

    def step(R, H, A, t):
        if t == n:
            finish(R, H, A)
            return
        B = w_hh @ A
        B[:, t] += w_hi
        pieces = [(R, H, np.ones(d, dtype=bool))]
        for j in range(d):
            nxt = []
            for R2, H2, act in pieces:
                if np.abs(B[j]).max() < 1e-14:
                    a2 = act.copy(); a2[j] = False; nxt.append((R2, H2, a2)); continue  # noqa: E702
                pl, mi = E.cut(R2, H2, B[j], ops)
                if pl is not None:
                    nxt.append((pl[0], pl[1], act))
                if mi is not None:
                    a2 = act.copy(); a2[j] = False; nxt.append((mi[0], mi[1], a2))  # noqa: E702
            pieces = nxt
        for R2, H2, act in pieces:
            step(R2, H2, B * act[:, None], t + 1)

    for R, H in E.orthants(n):
        step(R, H, np.zeros((d, n)), 0)


run(load("2nd_argmax_16_4_1073741824_bo5.pth"))
r = np.array(rec)
print(f"regions {len(r)}, total mass {r[:, 0].sum():.6f}, wrong mass {r[:, 1].sum():.6e}, accuracy {1 - r[:, 1].sum():.8f}")
for lab, key in [("by total mass", r[:, 0]), ("by wrong mass (oracle)", r[:, 1]), ("by wrong share", r[:, 1] / np.maximum(r[:, 0], 1e-300))]:
    o = np.argsort(-key)
    cw = np.cumsum(r[o, 1]) / r[:, 1].sum()
    cm = np.cumsum(r[o, 0])
    print(lab + ": share of wrong mass in the top " + ", ".join(f"{p}%: {cw[int(len(r) * p / 100) - 1]:.3f}" for p in (1, 5, 10, 20, 50)))
print("regions with any wrong mass:", int((r[:, 1] > 0).sum()))

"""Counterfactual scan: does partial switching decide whether a number after a record beats the previous largest?
Lists: previous largest P at a (P > 0.3), record M at r = a + 1..., (M between 1.2 P and 2.5 P), then x at r + k
(k = 1 or 2) set to s P; every other number set to -0.5 (so P is the second-largest unless x beats it). Final scores
at x's position and at P's position, normal and with the clipping of the four undone at x's step only. The right
answer flips from P to x at s = 1. Reported per s: share of lists naming x, naming P, and the median of
score[x] - score[P]. 20,000 lists per placement; placements (a, r, k) = (1, 3, 1), (2, 4, 1), (1, 3, 2), (2, 4, 2)."""
import numpy as np
from common import WI, WH, WO, C, lists, run
Xa = lists(8, 400_000)


def run_undo(Xs, j):
    def hook(t, pre, hn, hprev):
        if t == j: hn = hn.copy(); hn[:, C] = pre[:, C]
        return hn
    return run(Xs, hook=hook)[0]


for a, r, k in ((1, 3, 1), (2, 4, 1), (1, 3, 2), (2, 4, 2), (1, 5, 1)):
    P = np.abs(Xa[:, 0]) * 0.8 + 0.3; M = P * (1.2 + 1.3 * np.abs(Xa[:, 1]) / 2.5)
    ok = M < 2.5 * P
    X0 = np.full((400_000, 10), -0.5); X0[:, a] = P; X0[:, r] = M; X0 = X0[ok][:20000]; Pv = X0[:, a]; Mv = X0[:, r]
    j = r + k
    print(f"== P at {a}, record at {r}, x at {j}: median M/P {np.median(Mv / Pv):.2f}")
    print("   s = x/P | normal: names x, names P, median score[x]-score[P] | clip undone at x's step: names x, names P, median")
    for s in (0.8, 0.9, 0.95, 0.98, 1.0, 1.02, 1.05, 1.1, 1.2, 1.4):
        X1 = X0.copy(); X1[:, j] = np.minimum(s * Pv, 0.99 * Mv)
        out = []
        for L in (run(X1)[0], run_undo(X1, j)):
            pr = L.argmax(1); out.append(f"{np.mean(pr == j):.2f} {np.mean(pr == a):.2f} {np.median(L[:, j] - L[:, a]):+7.2f}")
        print(f"   {s:4.2f} | " + " | ".join(out))

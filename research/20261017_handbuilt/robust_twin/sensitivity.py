"""Which neurons make a hand-built twin fragile? Per neuron, the first-order sensitivity of the lead to noise of 1% of
the neuron's typical active value at every step: median over lists of sqrt(sum_t (g_{t,n} typ_n)^2) x 0.01 / lead, from
one backward pass (g as in gao/recurrent_condition.py). Above 1, noise of that size on that neuron alone typically flips
the answer. Networks: the trained network; v18 at trained scale (g = a = w = 10, lam 10, vote gain 10, tie-breaker 1% of
a vote; 92%)."""
import sys
import numpy as np
from algzoo.handbuilt_v18 import build_v18
import simlib

NAMES18 = ["store", "detector", "keep", "flip", "clock0", "clock1", "clock2", "flag"] + [f"mem{j}" for j in range(8)]
def sens(P, X):
    P = (np.asarray(P[0], float).reshape(-1), np.asarray(P[1], float), np.asarray(P[2], float)); r = np.arange(len(X))
    L, Pre, H = simlib.run(X, params=P); k = L.argmax(1); j = np.argsort(L, 1)[:, -2]; lead = L[r, k] - L[r, j]
    g = P[2][k] - P[2][j]; Gt = np.zeros_like(H)
    for t in range(9, -1, -1):
        Gt[:, t] = g; g = (g * (Pre[:, t] > 0)) @ P[1]
    typ = np.array([np.median(H[:, :, n][H[:, :, n] > 0]) if (H[:, :, n] > 0).any() else 0.0 for n in range(len(P[0]))])
    act = H > 0
    s = np.sqrt(((Gt * act * typ) ** 2).sum(1)) * 0.01 / np.maximum(lead, 1e-12)[:, None]
    return np.median(s, 0), typ
def main():
    X = simlib.lists(71, 20_000)
    for name, P, names in (("trained", (simlib.WI, simlib.WH, simlib.WO), [str(n) for n in range(len(P[0]))]),
                           ("v18 trained scale", build_v18(g=10, a=10, w=10, lam=10.0, c=-9.0, eps=0.1).params, NAMES18)):
        s, typ = sens(P, X)
        order = np.argsort(-s)
        print(f"{name}: per-neuron sensitivity to 1% noise (median over lists, relative to the lead; typical value):")
        print("   " + "; ".join(f"{names[n]} {s[n]:.2f} ({typ[n]:.2g})" for n in order))


if __name__ == "__main__":
    main()

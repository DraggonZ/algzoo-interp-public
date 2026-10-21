"""Exact two-spike comparator coefficients of degraded copies against the base model, by age of the max.

For each model: probability-weighted mean of a (coefficient on the max) and b (on the 2nd) per comparator neuron
and age of the max, over angle cells with m2 > 0, plus per-age two-spike accuracy. Prints the copies' deviation
from the base model.
"""
import numpy as np

from algzoo.bench import load_bench
from algzoo.core import example
from algzoo.exact_z2 import angle_mass, pair_intervals, correct_intervals

C = [1, 2, 4, 6, 7, 8]


def table(params):
    T = np.zeros((len(C), 10, 2))
    Wt = np.zeros(10)
    acc = np.zeros(10)
    mass = np.zeros(10)
    ops = [0]
    for p1 in range(10):
        for p2 in range(10):
            if p1 == p2:
                continue
            cells = pair_intervals(params, p1, p2, ops)
            for lo, hi in correct_intervals(params, p1, p2, cells, ops):
                acc[9 - p1] += angle_mass(lo, hi)
            for a, b, A in cells:
                mass[9 - p1] += angle_mass(a, b)
                lo = max(a, 0.0)
                if b <= lo:
                    continue
                m = angle_mass(lo, b)
                T[:, 9 - p1] += m * A[C]
                Wt[9 - p1] += m
    return T / Wt[None, :, None], acc / mass


if __name__ == "__main__":
    names, acc, models = load_bench(("mult",))
    Tb, ab = table(example())
    np.set_printoptions(linewidth=200, precision=2, suppress=True)
    print("base two-spike accuracy by age of max:", ab)
    for nm in ["mult0.02_18", "mult0.02_3", "mult0.01_14", "mult0.02_8"]:
        Tc, ac = table(models[list(names).index(nm)])
        print(f"\n{nm}: two-spike accuracy by age of max: {ac}")
        print("  relative change of a (coef on max) per neuron (rows n1, n2, n4, n6, n7, n8) by age 0..9:")
        print((Tc[:, :, 0] - Tb[:, :, 0]) / np.maximum(np.abs(Tb[:, :, 0]), 1.0))
        print("  change of b (coef on 2nd), absolute:")
        print(Tc[:, :, 1] - Tb[:, :, 1])

"""Grundmann–Möller rule on the standard simplex {μ >= 0, Σμ <= 1} in R^m, degree 2s+1.

Returns barycentric nodes (Q x (m+1)) and weights summing to the simplex volume 1/m!.
"""
import itertools
import math

import numpy as np


def compositions(total, parts):
    """All tuples of `parts` nonnegative integers summing to `total`."""
    for c in itertools.combinations(range(total + parts - 1), parts - 1):
        prev, out = -1, []
        for x in c:
            out.append(x - prev - 1)
            prev = x
        out.append(total + parts - 1 - prev - 1)
        yield out


def gm_rule(m, s):
    d = 2 * s + 1
    nodes, weights = [], []
    for i in range(s + 1):
        w = (-1) ** i * 2.0 ** (-2 * s) * (d + m - 2 * i) ** d / (math.factorial(i) * math.factorial(d + m - i))
        for beta in compositions(s - i, m + 1):
            nodes.append([(2 * b + 1) / (d + m - 2 * i) for b in beta])
            weights.append(w)
    return np.array(nodes), np.array(weights)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    for m in (2, 3, 4):
        for s in (1, 2):
            X, w = gm_rule(m, s)
            # exact integral of a monomial μ^a over the simplex: Π a_i! / (m + Σa)!
            worst = 0.0
            for _ in range(30):
                a = rng.integers(0, 2 * s + 2, size=m)
                if a.sum() > 2 * s + 1:
                    continue
                exact = np.prod([math.factorial(k) for k in a]) / math.factorial(m + a.sum())
                approx = (w * np.prod(X[:, :m] ** a, axis=1)).sum()
                worst = max(worst, abs(approx - exact) / exact)
            print(f"m={m} degree {2*s+1}: {len(w)} points, weights sum {w.sum():.6f} (1/m! = {1/math.factorial(m):.6f}), worst rel. error on monomials {worst:.1e}")

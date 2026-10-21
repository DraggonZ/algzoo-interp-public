"""r3a look: chained fair-vote push against the unchained joint solution (r2a-2).

Chained: covered rows are described in increasing order. Row k is the first row of the least-squares solution of the
push relations from ARC's true row k-1 (already described) up to the next uncovered anchor above k (rows in between
are nuisance unknowns). With no anchor above, the relations run to row LAST and the top is free.
Quick net = the scorer's net without KL and without STRUCTURE_BITS (share 0.001, one constant).

Usage: python -u r3a_quick.py
"""
import itertools
import numpy as np
import score_memory_rule as S
from r2a_joint import solve, quick, TGT, SRC, POS, A

WO = S.WO
n = len(TGT)
B = np.zeros((n, n)); B[POS, :] = A          # x @ B = push of x (x over TGT columns)


def bvp_first(left, right, m, g):
    """Unknowns x_1..x_m; relations x_1 = g left B, x_{i+1} = g x_i B, and (if right is not None) right = g x_m B.
    Returns x_1."""
    rows, rhs = [], []
    for i in range(m + (right is not None)):
        M = np.zeros((n, m * n)); r = np.zeros(n)
        if i < m:
            M[:, i * n:(i + 1) * n] += np.eye(n)
        else:
            r += right
        if i == 0:
            r += g * left @ B
        else:
            M[:, (i - 1) * n:i * n] -= g * B.T
        rows.append(M); rhs.append(r)
    x = np.linalg.lstsq(np.concatenate(rows), np.concatenate(rhs), rcond=None)[0]
    return x[:n]


def chained(rows, g, last=9):
    out = []
    for k in rows:
        above = [a for a in range(k + 1, last + 1) if a not in rows]
        if above:
            a = above[0]; out.append(bvp_first(WO[k - 1, TGT], WO[a, TGT], a - k, g))
        else:
            out.append(bvp_first(WO[k - 1, TGT], None, last - k + 1, g))
    return np.array(out)


def errs(rows, P):
    return np.round([np.linalg.norm(p - WO[t, TGT]) / np.linalg.norm(WO[t, TGT]) for t, p in zip(rows, P)], 3)


if __name__ == "__main__":
    G = (0.86, 0.89, 0.92, 0.95)
    base = [1, 2, 3, 4, 6, 7, 8]
    for name, f in (("unchained", lambda r, g: solve(r, g, 9)), ("chained", chained)):
        q, g = max((quick(base, f(base, g)), g) for g in G)
        print(f"{name:10s} rows {base}: quick {q:+.1f} g {g} errs {errs(base, f(base, g))}")
    res = []
    for k in range(3, 10):
        for rows in itertools.combinations(range(1, 10), k):
            rows = list(rows)
            q, g = max((quick(rows, chained(rows, g)), g) for g in G)
            res.append((q, rows, g))
    res.sort(reverse=True)
    for q, rows, g in res[:15]:
        print(f"chained quick {q:+.1f}  rows {rows}  g {g}  errs {errs(rows, chained(rows, g))}")

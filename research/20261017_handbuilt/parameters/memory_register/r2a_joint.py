"""r2a look: covered readout rows as the joint least-squares solution of the one-step push relations
row(t+1)[TGT] = g * row(t)[SRC] @ WH[SRC, TGT], t = 0 .. LAST-1, with ARC's uncovered rows as anchors.
Quick net = the scorer's net without the KL run and without STRUCTURE_BITS (share 0.001, one constant).

Usage: python -u r2a_joint.py
"""
import itertools
import numpy as np
import score_memory_rule as S

WH, WO = S.WH, S.WO
SRC = [i for i in range(16) if i not in (4, 5, 9)]
TGT = [0, 3, 10, 11, 12, 13, 14, 15, 1, 6, 7, 8, 2, 5]
POS = [TGT.index(i) for i in SRC]
A = WH[np.ix_(SRC, TGT)]


def solve(rows, g, last):
    """Return predicted covered rows (len(rows) x len(TGT))."""
    n = len(TGT); idx = {k: j for j, k in enumerate(rows)}
    eqs, rhs = [], []
    for t in range(last):
        if t not in idx and t + 1 not in idx:
            continue
        # residual: x_{t+1} - g x_t[SRC] A  (n equations)
        M = np.zeros((n, len(rows) * n)); r = np.zeros(n)
        if t + 1 in idx:
            j = idx[t + 1]; M[:, j * n:(j + 1) * n] += np.eye(n)
        else:
            r -= WO[t + 1, TGT]
        if t in idx:
            j = idx[t]; Bm = np.zeros((n, n)); Bm[POS, :] = g * A
            M[:, j * n:(j + 1) * n] -= Bm.T
        else:
            r += g * WO[t, SRC] @ A
        eqs.append(M); rhs.append(r)
    x = np.linalg.lstsq(np.concatenate(eqs), np.concatenate(rhs), rcond=None)[0]
    return x.reshape(len(rows), n)


def quick(rows, pred):
    cov = np.array([S.IDX[f"{i}->out{k}"] for k in rows for i in TGT])
    wr = pred.ravel()
    m = np.zeros(432); sd = S.s_prior.copy(); m[cov] = wr
    n = len(TGT)
    for j in range(len(rows)):
        ii = cov[j * n:(j + 1) * n]
        sd[ii] = np.sqrt(((wr[j * n:(j + 1) * n] - S.w_flat[ii]) ** 2).mean())
    return S.LOOKUP - (6 + 6 * (len(rows) + 1) + S.weight_bits(m, sd, S.BUDGET - 0.001))


if __name__ == "__main__":
    res = []
    for last in (8, 9):
        for k in range(3, 9):
            for rows in itertools.combinations(range(1, 9), k):
                rows = list(rows)
                # every covered run must touch an uncovered anchor (row 0 is always uncovered)
                g_best = max((quick(rows, solve(rows, g, last)), g) for g in (0.86, 0.89, 0.92))
                res.append((g_best[0], last, rows, g_best[1]))
    res.sort(reverse=True)
    for q, last, rows, g in res[:25]:
        err = [np.linalg.norm(p - WO[t, TGT]) / np.linalg.norm(WO[t, TGT]) for t, p in zip(rows, solve(rows, g, last))]
        print(f"quick {q:+.1f}  last {last}  rows {rows}  g {g}  rel errs {np.round(err, 3)}")

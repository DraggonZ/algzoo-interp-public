"""r3a look 3: where the zero-column-sum row pays. Quick net (scorer bits without KL and without STRUCTURE_BITS,
share 0.001, one constant) for push rows (chained, from ARC's row below, 14 columns) plus an optional sum row
(all 16 columns, minus the sum of ARC's other nine rows).

Usage: python -u r3a_look3.py
"""
import numpy as np
import score_memory_rule as S
from r3a_quick import B, TGT

WO = S.WO


def quick(push_rows, sum_row, g):
    names, vals = [], []
    for k in push_rows:
        names += [f"{i}->out{k}" for i in TGT]; vals += list(g * WO[k - 1, TGT] @ B)
    if sum_row is not None:
        names += [f"{i}->out{sum_row}" for i in range(16)]; vals += list(-WO[[k for k in range(10) if k != sum_row]].sum(0))
    cov = np.array([S.IDX[n] for n in names]); vals = np.array(vals)
    m = np.zeros(432); sd = S.s_prior.copy(); m[cov] = vals
    rws = np.array([S.row_of(n) for n in names])
    for r in set(rws):
        ii = cov[rws == r]; sd[ii] = np.sqrt(((vals[rws == r] - S.w_flat[ii]) ** 2).mean())
    return S.LOOKUP - (6 + 6 * (len(set(rws)) + 1) + S.weight_bits(m, sd, S.BUDGET - 0.001))


if __name__ == "__main__":
    for push_rows, sum_row in (([], 9), ([], 0), ([], 5), (range(1, 9), None), (range(1, 9), 9), (range(2, 10), 0),
                               (range(2, 9), 0), (range(1, 10), None), ([1, 2, 3, 4, 7, 8, 9], 5),
                               ([1, 2, 3, 4, 7, 8], 5)):
        q, g = max((quick(list(push_rows), sum_row, g), g) for g in np.arange(0.86, 0.97, 0.02))
        print(f"push {list(push_rows)}  sum row {sum_row}: quick {q:+.1f}  g {g:.2f}")
    print("sum-row relative errors by row:",
          np.round([np.linalg.norm(WO[k] + WO[[j for j in range(10) if j != k]].sum(0)) / np.linalg.norm(WO[k])
                    for k in range(10)], 2))

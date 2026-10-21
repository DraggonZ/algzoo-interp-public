"""r2a quick look: per-row error of one-sided (forward push) and two-sided (forward push from row t-1 plus row t+1 as
row t pushed) predictions of readout rows, and the scorer's net without the KL run (share 0.001, one constant).

Usage: python -u r2a_quick.py
"""
import math
import numpy as np
import score_memory_rule as S

WI, WH, WO = S.WI, S.WH, S.WO
SRC = [i for i in range(16) if i not in (4, 5, 9)]


def push_mat(tgt):
    return WH[np.ix_(SRC, SRC)], WH[np.ix_(SRC, tgt)]


def predict(t, tgt, mode, g):
    """Row t over columns tgt. mode 'f': g * row(t-1)[SRC] @ WH[SRC, tgt];
    'b': solve g * r[SRC] @ WH[SRC, tgt] = row(t+1)[tgt]; 'fb': least squares of both."""
    A = WH[np.ix_(SRC, tgt)]                     # SRC x tgt
    f = g * WO[t - 1, SRC] @ A
    pos = [tgt.index(i) for i in SRC]            # SRC columns inside tgt
    # backward: unknown r (len tgt); equations g * r[pos] @ WH[SRC, tgt] = WO[t+1, tgt]
    M_b = np.zeros((len(tgt), len(tgt)))
    M_b[pos, :] = g * A                           # r @ M_b = b
    b = WO[t + 1, tgt]
    if mode == "f":
        return f
    if mode == "b":
        return np.linalg.lstsq(M_b.T, b, rcond=None)[0]
    M = np.concatenate([np.eye(len(tgt)), M_b], 1)  # r @ M = [f, b]
    y = np.concatenate([f, b])
    return np.linalg.lstsq(M.T, y, rcond=None)[0]


def net(rows, tgt, preds):
    cov = np.array([S.IDX[f"{i}->out{k}"] for k in rows for i in tgt])
    wr = np.concatenate(preds)
    m = np.zeros(432); sd = S.s_prior.copy(); m[cov] = wr
    for j, k in enumerate(rows):
        ii = cov[j * len(tgt):(j + 1) * len(tgt)]
        sd[ii] = np.sqrt(((wr[j * len(tgt):(j + 1) * len(tgt)] - S.w_flat[ii]) ** 2).mean())
    left = S.weight_bits(m, sd, S.BUDGET - 0.001)
    return S.LOOKUP - (6 + 6 * (len(rows) + 1) + left)


if __name__ == "__main__":
    T14 = [0, 3, 10, 11, 12, 13, 14, 15, 1, 6, 7, 8, 2, 5]
    T16 = list(range(16))
    for tgt, name in [(T14, "14 cols"), (T16, "16 cols")]:
        print(f"== {name}")
        for t in range(1, 9):
            row = WO[t, tgt]; nr = np.linalg.norm(row)
            errs = {md: np.linalg.norm(predict(t, tgt, md, 0.92) - row) / nr for md in ["f", "b", "fb"]}
            print(f"  row {t}: rel err fwd {errs['f']:.3f}  back {errs['b']:.3f}  both {errs['fb']:.3f}")
    for tgt, name in [(T14, "14"), (T16, "16")]:
        for rows in [[1, 2, 4, 5, 7], [1, 4, 7], [1, 3, 5, 7], [2, 5, 7], [1, 2, 4, 5, 7, 8], [1, 3, 5, 7], [2, 4, 6, 8]]:
            for md in ["f", "fb"]:
                if md == "f" and False:
                    continue
                best = max((net(rows, tgt, [predict(t, tgt, md, g) for t in rows]), g)
                           for g in np.linspace(0.8, 1.1, 31))
                print(f"cols {name} rows {rows} mode {md}: quick net (minus structure) {best[0]:+.1f} at g {best[1]:.2f}")

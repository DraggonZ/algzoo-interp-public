"""r3a look 2: the plain chain (row k = g * ARC's row k-1 pushed one step, rows 1-9) and two variants:
(1) the push's residual persisting: row k = g row_{k-1} B + rho * e_{k-1} (or rho * e_{k-1} B), with
    e_{k-1} = ARC row k-1 - g ARC row k-2 B (known in a chain: both rows are described before row k);
(2) wider rows: also the detector's and late gate's columns, pushed through all 16 neurons.
Quick net as r2a_joint.quick (no KL, no structure, one constant; a second constant would cost about 8 more bits).

Usage: python -u r3a_look2.py
"""
import numpy as np
import score_memory_rule as S
from r3a_quick import B, TGT, errs, n
from r2a_joint import quick

WO = S.WO
G = np.arange(0.84, 1.001, 0.01)


def plain(rows, g):
    return np.array([g * WO[k - 1, TGT] @ B for k in rows])


def persist(rows, g, rho, pushed):
    out = []
    for k in rows:
        p = g * WO[k - 1, TGT] @ B
        if k >= 2:
            e = WO[k - 1, TGT] - g * WO[k - 2, TGT] @ B
            p = p + rho * (e @ B if pushed else e)
        out.append(p)
    return np.array(out)


if __name__ == "__main__":
    for rows in ([1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 5, 6, 7, 8]):
        q, g = max((quick(rows, plain(rows, g)), g) for g in G)
        print(f"plain rows {rows}: quick {q:+.1f} g {g:.2f} errs {errs(rows, plain(rows, g))}")
    rows = list(range(1, 10))
    for pushed in (False, True):
        best = max((quick(rows, persist(rows, g, r, pushed)), g, r) for g in G[::2] for r in np.arange(-0.6, 0.61, 0.1))
        q, g, r = best
        print(f"persist pushed={pushed}: quick {q:+.1f} g {g:.2f} rho {r:+.1f} errs {errs(rows, persist(rows, g, r, pushed))}")
    # the residual's own structure: cosine between consecutive residuals, raw and pushed
    g = 0.92
    E = {k: WO[k, TGT] - g * WO[k - 1, TGT] @ B for k in range(1, 10)}
    cos = lambda a, b: a @ b / np.linalg.norm(a) / np.linalg.norm(b)
    print("cos(e_k, e_{k-1}):   ", np.round([cos(E[k], E[k - 1]) for k in range(2, 10)], 2))
    print("cos(e_k, e_{k-1} B): ", np.round([cos(E[k], E[k - 1] @ B) for k in range(2, 10)], 2))
    print("residual by column (rms over rows 1-8, / row rms):")
    R = np.array([E[k] for k in range(1, 9)]); W = np.array([WO[k, TGT] for k in range(1, 9)])
    for j, c in enumerate(TGT):
        print(f"  col {c:2d}: resid {np.sqrt((R[:, j] ** 2).mean()):.2f}  weight {np.sqrt((W[:, j] ** 2).mean()):.2f}  "
              f"mean resid {R[:, j].mean():+.2f}")
    # wider rows: all 16 columns through the all-on block (and through the on-pattern block with 4, 9 columns kept)
    full = list(range(16))
    cov = lambda rows: np.array([S.IDX[f"{i}->out{k}"] for k in rows for i in full])
    def quick16(rows, P):
        c = cov(rows); m = np.zeros(432); sd = S.s_prior.copy(); m[c] = P.ravel()
        for j in range(len(rows)):
            ii = c[j * 16:(j + 1) * 16]; sd[ii] = np.sqrt(((P[j] - S.w_flat[ii]) ** 2).mean())
        return S.LOOKUP - (6 + 6 * (len(rows) + 1) + S.weight_bits(m, sd, S.BUDGET - 0.001))
    q, g = max((quick16(rows, np.array([g * WO[k - 1] @ S.WH for k in rows])), g) for g in G)
    print(f"16 columns, all-on block: quick {q:+.1f} g {g:.2f}")

"""Where is the trained margin at a near-tie overtaking built? Lists (as bridge/margin_source.py): a record at r (the
final second-largest, above 0.3) overtaken at q by a number under 5% larger that stays the largest. Margin
M = score[r] - score[q]. Two nudges: 'level' (x_r and x_q both up, jump fixed) and 'jump' (x_q up).
(1) Injection split, exact for the run's on/off pattern: dM/dx_t = sum_n a_n(t) [n on at t] w_n, with a(t) = dM/dh(t)
    by backpropagation through the run's pattern; reported per neuron for the inputs at r and at q.
(2) Carriage split: at every step t >= q the whole effect passes through h(t): dM = sum_n a_n(t) dh_n(t) (dh by finite
    difference); reported per group at steps q-1 .. 9.
Medians over lists, per unit of the nudge. 3,000,000 lists scanned (seed 17); placements (r, q) = (1, 4), (2, 5), (3, 6), (4, 7)."""
import numpy as np
from common import WI, WH, WO, C, GROUPS, lists, run

X = lists(17, 3_000_000)
G = {"hiding": [1, 6, 7, 8], "store": [2], "detector": [4], "keep gate": [5], "late gate": [9], "memory": [0, 3, 10, 11, 12, 13, 14, 15]}


def backprop(P, rr, qq):
    """a(t) = dM/dh(t) for t = 0..9 through the pattern of pre-activations P (B, 10, 16)."""
    B = len(P); a = np.zeros((B, 10, 16)); a[:, 9] = WO[rr] - WO[qq]
    for t in range(9, 0, -1):
        a[:, t - 1] = (a[:, t] * (P[:, t] > 0)) @ WH
    return a


def sel(rr, qq):
    pre = np.maximum(X[:, :rr].max(1), 0) if rr > 0 else np.zeros(len(X))
    ok = (X[:, rr] > pre) & (X[:, rr] > 0.3) & (X[:, rr + 1:qq].max(1) < X[:, rr]) & (X[:, qq] > X[:, rr]) & (X[:, qq] < 1.05 * X[:, rr])
    if qq < 9: ok &= X[:, qq + 1:].max(1) < X[:, rr]
    return X[ok]


eps = 1e-4


def main():
    for rr, qq in ((1, 4), (2, 5), (3, 6), (4, 7)):
        Xs = sel(rr, qq); L0, P0, H0 = run(Xs); M0 = L0[:, rr] - L0[:, qq]; a = backprop(P0, rr, qq)
        print(f"== record at {rr}, overtaken at {qq} by under 5% (n={len(Xs)}): median margin {np.median(M0):.2f}, right {np.mean(L0.argmax(1) == rr):.3f}")
        for lab, pos in (("level (both up, jump fixed)", (rr, qq)), ("jump (new largest up)", (qq,))):
            X1 = Xs.copy()
            for p_ in pos: X1[:, p_] += eps
            L1, P1, H1 = run(X1); dM = ((L1[:, rr] - L1[:, qq]) - M0) / eps
            inj = {}
            for p_ in pos:
                v = a[:, p_] * (P0[:, p_] > 0) * WI   # per neuron
                inj[p_] = {g: np.median(v[:, idx].sum(1)) for g, idx in G.items()}
            tot_inj = sum((a[:, p_] * (P0[:, p_] > 0) * WI).sum(1) for p_ in pos)
            print(f"  {lab}: dM per unit, median {np.median(dM):+7.2f} (backprop {np.median(tot_inj):+7.2f})")
            for p_ in pos:
                print(f"     input at step {p_} enters through: " + ", ".join(f"{g} {v:+7.1f}" for g, v in inj[p_].items()))
            dH = (H1 - H0) / eps
            for t in range(qq - 1, 10):
                parts = {g: np.median((a[:, t][:, idx] * dH[:, t][:, idx]).sum(1)) for g, idx in G.items()}
                tot = np.median((a[:, t] * dH[:, t]).sum(1))
                print(f"     carried at step {t}: total {tot:+7.2f} | " + ", ".join(f"{g} {v:+7.1f}" for g, v in parts.items()))


if __name__ == "__main__":
    main()

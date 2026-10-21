"""Exact (forward-mode, within each run's on/off pattern) split of how the scores at the old record's position r and
the new largest's position q depend on the old record x_r and the new largest x_q, on near-tie overtaking lists
(margin_route.sel). Means over lists (means add up; medians do not). For each step t from q to 9, the share carried by
each group, a_G(t) . dh_G(t), where a(t) = d score / d h(t); shares at one step sum to the total. Nudges: x_q alone
(the new largest), x_r alone (the old record), both (level, jump fixed)."""
import numpy as np
from common import WI, WH, WO, lists, run
from margin_route import sel, G


def fwd(P, dx):
    """dh(t) for an input change dx (B, 10), through the run's pattern."""
    B = len(P); dh = np.zeros((B, 16)); out = []
    for t in range(10):
        dh = (P[:, t] > 0) * (dh @ WH.T + dx[:, t:t + 1] * WI); out.append(dh)
    return np.stack(out, 1)


def adj(P, wvec):
    B = len(P); a = np.zeros((B, 10, 16)); a[:, 9] = wvec
    for t in range(9, 0, -1): a[:, t - 1] = (a[:, t] * (P[:, t] > 0)) @ WH
    return a


def main():
    for rr, qq in ((1, 4), (2, 5), (3, 6), (4, 7)):
        Xs = sel(rr, qq); L0, P0, _ = run(Xs); B = len(Xs)
        print(f"== record at {rr}, overtaken at {qq} (n={B}); mean margin {np.mean(L0[:, rr] - L0[:, qq]):.2f}")
        for lab, pos in (("x_q alone", [qq]), ("x_r alone", [rr]), ("level (both)", [rr, qq])):
            dx = np.zeros((B, 10)); dx[:, pos] = 1.0; dH = fwd(P0, dx)
            for sname, wv in (("score[r]", WO[rr]), ("score[q]", WO[qq]), ("margin", WO[rr] - WO[qq])):
                a = adj(P0, np.broadcast_to(wv, (B, 16)))
                tot = np.mean((a[:, 9] * dH[:, 9]).sum(1))
                rows = []
                for t in (qq + 1, qq + 2, 9):
                    if t > 9: continue
                    rows.append(f"t={t}: " + " ".join(f"{g[:4]} {np.mean((a[:, t][:, idx] * dH[:, t][:, idx]).sum(1)):+6.1f}" for g, idx in G.items()))
                print(f"  {lab:13s} {sname:8s} total {tot:+6.1f} | " + " || ".join(rows))


if __name__ == "__main__":
    main()

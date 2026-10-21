"""r8b look 2: value of each uncovered hiding-row weight if stated exactly (alone, added to the current best), and of
candidate zero-constant statements. Diagnostic only."""
import numpy as np
import r8b_chainx as X
S, WI, WH = X.S, X.WI, X.WH
cur, _ = X.rule_weights(X.CURRENT)
g0 = X.gain(cur)
H = [1, 6, 7, 8]
RS = WH[4, 2] / WI[4]
unc = [f"{i}->{j}" for j in H for i in range(16) if f"{i}->{j}" not in cur]
w = lambda n: S.w_flat[S.IDX[n]]
for n in unc:
    p = dict(cur); p[n] = w(n)
    print(f"{n:6s} ARC {w(n):+.3f}  exact alone {X.gain(p) - g0:+5.1f}")


def blind(p, j, i=7):
    """weight i->j that makes row j blind, from the other three (the rule's value if covered, else ARC's)"""
    get = lambda k: p.get(f"{k}->{j}", WH[j, k])
    return -sum(get(k) * WI[k] for k in H if k != i) / WI[i]


def withblind(p):
    for j in (1, 6, 8): p[f"7->{j}"] = blind(p, j)
    return p


cands = {
    "4->1 = W12 W[2,4]": {"4->1": RS * WI[1] * WH[2, 4]},
    "A[1,6]=1": {"6->1": WI[1] / WI[6]},
    "A[8,6]=0": {"6->8": 0.0},
    "late gate exact 9->1,9->6": {"9->1": WH[1, 9], "9->6": WH[6, 9]},
}
for k, c in cands.items():
    p = withblind({**cur, **c}); print(f"{k:28s} {X.gain(p) - g0:+5.1f}  err " + " ".join(f"{n}:{p[n] - w(n):+.3f}" for n in c))
p = withblind({**cur, **cands["4->1 = W12 W[2,4]"], **cands["A[1,6]=1"], **cands["A[8,6]=0"]})
print(f"three together {X.gain(p) - g0:+5.1f}")

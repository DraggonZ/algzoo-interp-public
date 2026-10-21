"""r8b look 4: price the planned zero-constant statements on top of the current best (chain explorer). Diagnostic."""
import numpy as np
import r8b_chainx as X
S, WI, WH = X.S, X.WI, X.WH
cur, _ = X.rule_weights(X.CURRENT)
g0 = X.gain(cur)
H = [1, 6, 7, 8]
RS = WH[4, 2] / WI[4]
w = lambda n: S.w_flat[S.IDX[n]]


def withblind(p):
    for j in (1, 6, 8):
        p[f"7->{j}"] = -sum(p.get(f"{k}->{j}", WH[j, k]) * WI[k] for k in H if k != 7) / WI[7]
    return p


S1 = {"4->1": RS * WI[1] * WH[2, 4]}                       # fresh store, fade ignored
S1f = {"4->1": RS * WI[1] * WH[2, 4] / WH[2, 2]}           # fresh store with fade
S2 = {"6->1": WI[1] / WI[6], "6->8": 0.0}                  # excess passing: 6's excess to 1 whole, none to 8
S3 = {"8->8": WH[8, 1]}                                     # 8 reads 1 and itself equally (raw)
for k, c in [("S1", S1), ("S1 with fade", S1f), ("S2", S2), ("S1+S2", {**S1, **S2}), ("S1+S2+S3", {**S1, **S2, **S3})]:
    p = withblind({**cur, **c})
    print(f"{k:14s} {X.gain(p) - g0:+5.1f}  err " + " ".join(f"{n}:{p[n] - w(n):+.3f}" for n in list(c) + ["7->1", "7->8"]))

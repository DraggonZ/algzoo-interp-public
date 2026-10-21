"""r8b look 5: the memory columns of rows 6 and 8 (and 1) taken from another row already described, scaled by input
weights (W[j,m] = w_j / w_k W[k,m]): which source neuron k fits best. Diagnostic only."""
import numpy as np
import r8b_chainx as X
S, WI, WH = X.S, X.WI, X.WH
cur, _ = X.rule_weights(X.CURRENT)
g0 = X.gain(cur)
M = [0, 3, 10, 11, 12, 13, 14, 15]
for j in (6, 8, 1):
    for k in (4, 7, 1, 6, 8):
        if k == j: continue
        pred = {f"{m}->{j}": WI[j] / WI[k] * WH[k, m] for m in M}
        e = np.array([pred[n] - WH[j, int(n.split('->')[0])] for n in pred])
        p = {**cur, **pred}
        print(f"row {j} memory from row {k}: rms err {np.sqrt((e ** 2).mean()):.4f}  gain change {X.gain(p) - g0:+5.1f}")

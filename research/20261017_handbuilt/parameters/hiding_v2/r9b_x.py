"""r9b look 1: candidate statements for row 7's hiding columns and row 8's reads of 1 and 8, priced in the chain
explorer (r8b_chainx: the scorer's rest bits at share 0.001, base constant at its default fit, 6 bits per covered
row, no structure) against rules/r8b_chain.py. Diagnostic only."""
import numpy as np
import r8b_chainx as X
S, WI, WH = X.S, X.WI, X.WH
H = [1, 6, 7, 8]
RS = WH[4, 2] / WI[4]
cur, _ = X.rule_weights(X.HERE / "rules" / "r8b_chain.py")
g0 = X.gain(cur)
print(f"r8b_chain gross {g0:.1f}")
X.row_report(cur)


def show(name, ch):
    p = {**cur, **ch}
    e = {n: ch[n] - WH[int(n.split('->')[1]), int(n.split('->')[0])] for n in ch}
    print(f"{name:55s} gain {X.gain(p) - g0:+6.1f}   " + " ".join(f"{n}:{v:+.3f}" for n, v in e.items()))
    return p


def blind(row, j, i=7):
    return -sum(row[k] * WI[k] for k in H if k != i) / WI[i]


# 1. row 8: 1->8 = 8->8 = store weight; 7->8 by blindness
r8 = {1: RS * WI[8], 6: 0.0, 8: RS * WI[8]}
c1 = {"1->8": r8[1], "8->8": r8[8], "7->8": blind(r8, 8)}
show("row8: 1,8 = store weight, 7 blind", c1)
# using ARC's 2->8 raw instead of RS w8 (not covered? it is covered by rule) -- skip

# 2. row 7 hiding columns
mir = {i: WI[7] / WI[4] * WH[4, i] for i in H}
w72 = WI[7] / WI[4] * WH[4, 2]
ca = {f"{i}->7": mir[i] + w72 * WH[2, i] for i in H}
show("row7: mirror + W72 * W[2,i] (fresh store)", ca)
pat = {1: 1, 6: 1, 7: -1, 8: -1}
cb = {f"{i}->7": mir[i] + RS * pat[i] * WI[7] / WI[i] for i in H}
show("row7: mirror + RS(1,1,-1,-1) ref units", cb)
for f in (1.5, 2.0, 2.2):
    show(f"row7: hiding cols = {f} x mirror", {f"{i}->7": f * mir[i] for i in H})
show("both 1 and 2b", {**c1, **cb})
show("both 1 and 2a", {**c1, **ca})
show("row7 hiding exact", {f"{i}->7": WH[7, i] for i in H})

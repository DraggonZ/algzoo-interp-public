"""r15a look 1: the current best's readout residual per row and column (rule minus ARC), and its price."""
import numpy as np, time
import r12a_x as X
t = time.time()
R = X.load("rules/r14a_minarrival.py")
net, d = X.price(R)
print("net %+.1f gain %s (%.0fs)" % (net, d["C"], time.time() - t))
WO = X.WO
P = np.zeros((10, 16))
for n, v in d["w"].items():
    s, tg = n.split("->")
    if tg.startswith("out"):
        P[int(tg[3:]), int(s)] = v
np.set_printoptions(precision=2, suppress=True, linewidth=200)
print("ARC readout\n", WO)
print("residual rule-ARC rows 1-9\n", (P - WO)[1:])
print("row rel err", np.linalg.norm((P - WO)[1:], axis=1) / np.linalg.norm(WO[1:], axis=1))
print("col sums ARC", WO.sum(0))
print("covered names", [n for n in R.COVERED if not n.split('->')[1].startswith('out')][:80])

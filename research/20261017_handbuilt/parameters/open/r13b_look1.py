"""r13b look 1: is the readout blind to the hiding neurons' input direction? Compare ARC's rows with r12a's pushed rows."""
import numpy as np
import r12a_x as X
R = X.load("rules/r12a_regpush.py")
WI, WH, WO = X.WI, X.WH, X.WO
HID = [1, 6, 7, 8]
g = 1.0177
P = np.zeros_like(WO); P[1:9] = g * R._REG
wi = WI[HID]
print("w_in hiding", wi.round(2))
for k in range(10):
    a = WO[k, HID]; p = P[k, HID]
    s = f"row {k}: ARC hid cols {a.round(2)} cos w_in {a @ wi / np.linalg.norm(a) / np.linalg.norm(wi):+.2f}"
    if 1 <= k <= 8:
        s += f" | push {p.round(2)} cos {p @ wi / np.linalg.norm(p) / np.linalg.norm(wi):+.2f} resid {(a - p).round(2)}"
    print(s)
for k in range(10):
    s = "%d cos(row, WI) ARC %+.2f" % (k, WO[k] @ WI / np.linalg.norm(WO[k]) / np.linalg.norm(WI))
    if 1 <= k <= 8:
        s += " push %+.2f" % (P[k] @ WI / np.linalg.norm(P[k]) / np.linalg.norm(WI))
    print(s)
E = WO[1:9] - P[1:9]
print("resid rms per column", np.sqrt((E ** 2).mean(0)).round(2))
print("ARC rms per column  ", np.sqrt((WO ** 2).mean(0)).round(2))
print("WI", WI.round(2))

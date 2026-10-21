"""r13b look 6: the residual of rows 1-8 after the zero-sum offset (r13b_gauge). Where is it, and along what?"""
import numpy as np
import torch as th
import r12a_x as X

R = X.load("rules/r13b_gauge.py")
WI, WH, WO = X.WI, X.WH, X.WO
net, d = X.price(R)
g = d["C"][0]
w = d["w"]
P = np.array([[float(w[f"{i}->out{k}"]) for i in range(16)] for k in range(1, 9)])
E = WO[1:9] - P
np.set_printoptions(precision=2, suppress=True, linewidth=160)
print("net", round(net, 1), "gain", g)
print("residual rows 1-8 (columns 0..15):")
print(E)
print("rms per column", np.sqrt((E ** 2).mean(0)))
U, S, Vt = np.linalg.svd(E)
print("singular values", S)
for j in range(3):
    v = Vt[j]
    cs = [v @ WO[r] / np.linalg.norm(WO[r]) for r in range(10)]
    print(f"sv{j}: dir {v} | row loadings {U[:, j] * S[j]}")
    print("     cos with ARC rows 0-9", np.array(cs))
# residual vs the previous row and vs the row itself (gain per row)
for k in range(1, 9):
    e = E[k - 1]
    print(k, "cos(e,row k) %+.2f cos(e,row k-1) %+.2f cos(e,row0) %+.2f cos(e,row9) %+.2f cos(e,row4) %+.2f" % tuple(
        e @ WO[r] / np.linalg.norm(e) / np.linalg.norm(WO[r]) for r in (k, k - 1, 0, 9, 4)))

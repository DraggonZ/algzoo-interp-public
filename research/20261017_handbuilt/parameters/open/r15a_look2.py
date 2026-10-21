"""r15a look 2: is the pushed rows' leftover error in directions the final state barely varies along?
Eigen-split of the push regression's Gram matrix G = E[h_10 h_10^T]; residual (ARC - rule, rows 1-8) per eigen-direction,
in weight units and in function units (sqrt(eigenvalue) x weight)."""
import numpy as np
import r12a_x as X

R = X.load("rules/r14a_minarrival.py")
net, d = X.price(R)
WO = X.WO
P = np.zeros((10, 16))
for n, v in d["w"].items():
    s, tg = n.split("->")
    if tg.startswith("out"):
        P[int(tg[3:]), int(s)] = v
E = (WO - P)[1:9]                      # leftover, rows 1-8
G = R._G
lam, U = np.linalg.eigh(G)
np.set_printoptions(precision=3, suppress=True, linewidth=200)
print("eigenvalues of G:", lam)
proj = E @ U                           # [8, 16]
print("leftover weight energy per eigen-direction (sum over rows 1-8):", (proj ** 2).sum(0))
print("share of leftover weight energy in the 4 smallest:", (proj[:, :4] ** 2).sum() / (proj ** 2).sum())
print("share in the 6 smallest:", (proj[:, :6] ** 2).sum() / (proj ** 2).sum())
print("ARC rows 1-8 energy per eigen-direction:", ((WO[1:9] @ U) ** 2).sum(0))
for i in range(5):
    print("eigvec %d (lam %.4f):" % (i, lam[i]), U[:, i])

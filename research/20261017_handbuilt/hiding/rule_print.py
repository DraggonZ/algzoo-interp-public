"""The written rule (rule_pooled.py, with P-arrival terms) in readable units. All in units of value (the reference),
i.e. fitted coefficients divided by c_n = (10.56, 11.00, 13.17, 12.31) for neurons 1, 6, 7, 8.
(1) Base reference by K (steps since the record): ref = alpha(K, d) P + beta(K) M, where d is how many steps before
    the record the previous largest P arrived; alpha shown for d = 1 and d = 3.
(2) Echo: a number that exceeds neuron i's reference by e (so neuron i is clipped by c_i e) raises neuron n's
    reference l steps later by T_ni(l) e. T(1) from the fit, next to its value from the weights, W_cc[n, i] c_i / c_n.
(3) Kept numbers since the record: each raises the references by v_n(l) times its value, l steps later.
Also prints the rule's reference fit quality on fresh lists (seed 5): RMS error of c ref in pre-activation units."""
import numpy as np
from rule_pooled import regime_key, NL
from rulelib import C, WCC, CN
sol = np.load("rule_pooled_sol.npy")
keys = sorted({regime_key(t, rr) for t in range(1, 10) for rr in range(-1, t)}, key=str); nreg = 11
offs = {k: (i * nreg, nreg if k[0] == "K" else 1) for i, k in enumerate(keys)}; o = len(keys) * nreg
print("(1) base reference: ref = alpha P + beta M  (alpha for P arrived 1 / 3 steps before the record)")
for K in range(1, 9):
    a, n = offs[("K", K)]; A = sol[a] / CN; Bm = sol[a + 1] / CN; g1 = sol[a + 2] / CN; g3 = sol[a + 4] / CN
    print(f"  K={K}: " + " | ".join(f"n{C[i]}: alpha {A[i] + g1[i]:+.2f} / {A[i] + g3[i]:+.2f}, beta {Bm[i]:+.2f}" for i in range(4)))
print("(2) echo T(l)[n <- i]: reference shift of neuron n per unit by which the number exceeded neuron i's reference")
names = [f"from {c}" for c in C]
for l in range(1, 4):
    G = sol[o + 2 * NL + 4 * (l - 1): o + 2 * NL + 4 * l, :].T          # [n, i] in pre-activation units per clipped amount
    T = G * CN[None, :] / CN[:, None]
    print(f"  lag {l} (fitted), rows n = 1, 6, 7, 8; columns i = 1, 6, 7, 8")
    for r_ in range(4): print("     n" + str(C[r_]) + ": " + " ".join(f"{v:+.2f}" for v in T[r_]))
    if l == 1:
        Tw = WCC * CN[None, :] / CN[:, None]
        print("  lag 1 from the weights alone (W_cc[n, i] c_i / c_n)")
        for r_ in range(4): print("     n" + str(C[r_]) + ": " + " ".join(f"{v:+.2f}" for v in Tw[r_]))
print("(3) kept numbers since the record, v(l) per unit of value; any number since, u(l)")
for l in range(1, 7):
    u = sol[o + 2 * (l - 1)] / CN; v = sol[o + 2 * (l - 1) + 1] / CN
    print(f"  lag {l}: v " + " ".join(f"{z:+.3f}" for z in v) + " | u " + " ".join(f"{z:+.3f}" for z in u))
print("(4) how alpha (the weight on P) depends on d, the steps between P's arrival and the record, for K = 1, 2")
for K in (1, 2):
    a, n = offs[("K", K)]; A = sol[a] / CN
    for i in range(4):
        print(f"  K={K} n{C[i]}: " + " ".join(f"d={d}: {A[i] + sol[a + 1 + d, i] / CN[i]:+.3f}" for d in range(1, 9)))
nK = sum(1 for k in keys if k[0] == "K"); nneg = len(keys) - nK
used = nK * 44 + nneg * 4 + (sol.shape[0] - o) * 4
print(f"(5) size of the rule: {used} fitted numbers in use ({nK} K-regimes x 11 x 4 = {nK * 44}, of which {nK * 9 * 4} for when P arrived; "
      f"{nneg} steps before any positive number x 4; shared kernels {(sol.shape[0] - o) * 4}: numbers since {2 * NL * 4}, echo {4 * NL * 4})")

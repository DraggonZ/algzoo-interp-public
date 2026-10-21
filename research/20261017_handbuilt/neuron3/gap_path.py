"""Q2/Q3: what the hiding-difference path into neuron 3 does to the final scores. Exact split within the real
on/off pattern: the path's share of score k from number p is sum_t G[t,k,3] * (W31 J1_{t-1,p} + W38 J8_{t-1,p}) x_p.
Compared with the number's total effect (W_oh J_9) and with all of neuron 3's inputs together."""
from n3lib import *

X = lists(8, 100_000); S, P, H, JP, JH = jac(X); G = back(P); n = len(X); r = np.arange(n)
o = X.argsort(1); L, Sd = o[:, -1], o[:, -2]
JHs = np.concatenate([np.zeros_like(JH[:, :1]), JH[:, :-1]], 1)          # J of h_{t-1}
gapJ = WH[3, 1] * JHs[:, :, 1] + WH[3, 8] * JHs[:, :, 8]                  # (n,T,p): per-unit gap term entering pre3_t
allJ = np.einsum("j,btjp->btp", WH[3], JHs); allJ[:, :, :] += 0
for t in range(10): allJ[:, t, t] += WI[3]                               # everything entering pre3_t
pathgap = np.einsum("btk,btp->bkp", G[:, :, :, 3], gapJ)                  # per unit: d score_k / d x_p via the gap path
path3 = np.einsum("btk,btp->bkp", G[:, :, :, 3], allJ)                    # via any write into neuron 3
tot = np.einsum("ki,bip->bkp", WO, JH[:, 9])
np.set_printoptions(precision=1, suppress=True, linewidth=160)
for nm, idx in (("second-largest", Sd), ("largest", L)):
    print(f"\n=== per unit of the {nm}, by its position: score at its own position, and mean over the other positions")
    print("pos   total-own  n3-own  gap-own  |  total-other  n3-other  gap-other  | (own minus mean other)  total  n3  gap")
    for p in range(10):
        m = idx == p
        def own(A): return A[m, p, p].mean()
        def oth(A): return (A[m][:, :, p].sum(1) - A[m, p, p]).mean() / 9
        print(f"{p:3d}  " + "  ".join(f"{z:8.1f}" for z in (own(tot), own(path3), own(pathgap), oth(tot), oth(path3), oth(pathgap)))
              + "   | " + "  ".join(f"{own(A) - oth(A):6.1f}" for A in (tot, path3, pathgap)))
print("\nper unit of the second-largest via the gap path, score at each position (rows = position of the second)")
print(np.array([pathgap[Sd == s][:, :, s].mean(0) for s in range(10)]))
print("\nper unit of the largest via the gap path, score at each position (rows = position of the largest)")
print(np.array([pathgap[L == s][:, :, s].mean(0) for s in range(10)]))

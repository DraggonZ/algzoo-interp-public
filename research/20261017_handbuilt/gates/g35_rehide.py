"""Does the keep gate's write after a kept number re-hide the largest's vote? Random lists (100,000, seed 35) whose
second-largest S came after the largest L, and L at least two steps before S. The keep gate is silenced for its receivers
only at the step after S (readout unchanged; S at 8 excluded so the readout of that step is not involved).
Measured: change in L's and S's final scores. Compared with how much S switched the hiding neurons off at its step:
for each hiding neuron, its value with x_S replaced by 0 minus its actual value (the hiding the arrival removed)."""
import numpy as np
from gatelib import WI, WH, lists, run

X = lists(35, 100_000); o = X.argsort(1); iL, iS = o[:, -1], o[:, -2]
sel = (iS > iL + 1) & (iS <= 7)
X = X[sel]; iL = iL[sel]; iS = iS[sel]; B = np.arange(len(X))
L0, P0, H0 = run(X)


def hook(t, hp, pre, h):
    m = (iS + 2) == t
    if m.any():
        h[m] = np.maximum(pre[m] - np.outer(hp[m, 5], WH[:, 5]), 0)
    return h


L1 = run(X, hook=hook)[0]
dL = L1[B, iL] - L0[B, iL]; dS = L1[B, iS] - L0[B, iS]
print(f"{len(X)} lists. Silencing the keep gate after S: L's score changes by {dL.mean():+.2f} (median {np.median(dL):+.2f}), S's by {dS.mean():+.2f} (median {np.median(dS):+.2f})")
ok0 = L0.argmax(1) == iS; ok1 = L1.argmax(1) == iS
print(f"  right: intact {100 * ok0.mean():.1f}%, silenced {100 * ok1.mean():.1f}%; silenced and wrong lists name L in {100 * (L1.argmax(1)[~ok1] == iL[~ok1]).mean():.1f}%")
# hiding removed by S's arrival
hs = H0[B, iS - 1]                                  # state before S
u = hs @ WH.T
act = np.maximum(u + X[B, iS][:, None] * WI, 0); zero = np.maximum(u, 0)
lost = (zero - act)[:, [1, 6, 7, 8]]
A = np.c_[lost, X[B, iS], np.ones(len(X))]
for name, y in (("L's change", dL), ("S's change", dS)):
    c = np.linalg.lstsq(A, y, rcond=None)[0]; e = y - A @ c
    print(f"  {name} ~ " + " ".join(f"{v:+.2f}*lost{j}" for v, j in zip(c[:4], (1, 6, 7, 8))) + f" {c[4]:+.2f}*S {c[5]:+.2f}  (variance matched {1 - e.var() / y.var():.2f})")
c = np.linalg.lstsq(A[:, 4:], dL, rcond=None)[0]; e = dL - A[:, 4:] @ c
print(f"  L's change from S's value alone: variance matched {1 - e.var() / dL.var():.2f}")
tot = lost.sum(1)
for lo, hi in ((0, 1), (1, 5), (5, 10), (10, 20), (20, 99)):
    m = (tot >= lo) & (tot < hi)
    print(f"  hiding removed by S (sum over 1,6,7,8) in [{lo},{hi}): {m.sum():6d} lists, L's change {dL[m].mean():+6.2f}, S's change {dS[m].mean():+6.2f}")

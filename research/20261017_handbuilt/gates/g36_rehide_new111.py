"""The re-hiding test of g35_rehide.py on new111's keep gate (neuron 10): random lists (100,000, seed 35) whose
second-largest S came at least two steps after the largest L (S at 7 or earlier); the keep gate is silenced for its
receivers only at the step after S. Change in L's and S's final scores."""
import numpy as np
from gatelib import lists, run, other_models

wi, wh, wo = other_models()["new111"]; wi = wi.reshape(-1); g = 10
X = lists(35, 100_000); o = X.argsort(1); iL, iS = o[:, -1], o[:, -2]
sel = (iS > iL + 1) & (iS <= 7)
X = X[sel]; iL = iL[sel]; iS = iS[sel]; B = np.arange(len(X))
L0 = run(X, params=(wi, wh, wo))[0]


def hook(t, hp, pre, h):
    m = (iS + 2) == t
    if m.any():
        h[m] = np.maximum(pre[m] - np.outer(hp[m, g], wh[:, g]), 0)
    return h


L1 = run(X, params=(wi, wh, wo), hook=hook)[0]
dL = L1[B, iL] - L0[B, iL]; dS = L1[B, iS] - L0[B, iS]
ok0 = L0.argmax(1) == iS; ok1 = L1.argmax(1) == iS
print(f"new111, {len(X)} lists: silencing the keep gate after S changes L's score by {dL.mean():+.2f} (median {np.median(dL):+.2f}), S's by {dS.mean():+.2f} (median {np.median(dS):+.2f})")
print(f"  right: intact {100 * ok0.mean():.1f}%, silenced {100 * ok1.mean():.1f}%; silenced and wrong lists name L in {100 * (L1.argmax(1)[~ok1] == iL[~ok1]).mean():.1f}%")

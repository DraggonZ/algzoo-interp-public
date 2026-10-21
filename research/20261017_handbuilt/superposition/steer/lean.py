"""Does the account predict the lean at near ties, pair by pair? Measured: for each (second's position p, third's q),
the share of near ties (third within 5% of the second) that the third wins. Predicted from the weights: on each
arrangement's canonical list the network is linear, scores = J x; with the others fixed, the third wins when
x_T / x_S > r* = (J[p,p] - J[q,p]) / (J[q,q] - J[p,q]) (the lead's two coefficients); the lean is 1 - r* (positive:
leans to the third), averaged over the largest's position. Score: correlation over the 90 pairs, and the predicted
win share from r* under the near-tie ratio distribution."""
import sys; sys.path.insert(0, "../level2/deduce")
import numpy as np
from harness import *
from case_maps import A, idx, C
J = jac(ARC, C).cpu().numpy()                                   # 720 x 10 x 10
lean = {}
for i, (l, p, q) in enumerate(A):
    a = J[i, p, p] - J[i, q, p]; b = J[i, q, q] - J[i, p, q]
    lean.setdefault((p, q), []).append(1 - a / b)
pred = {k: np.median(v) for k, v in lean.items()}
X = lists(8_000_000, seed=91); o = X.argsort(1, descending=True); v = X.sort(1, descending=True).values
near = (v[:, 1] > 0) & ((v[:, 1] - v[:, 2]) < 0.05 * v[:, 1]); a0 = run(ARC, X).argmax(1)
P, Q = o[:, 1][near].cpu().numpy(), o[:, 2][near].cpu().numpy(); win = (a0[near] == o[:, 2][near]).cpu().numpy()
ratio = (v[:, 2] / v[:, 1])[near].cpu().numpy()
meas, prd, shr = [], [], []
for (p, q), r in pred.items():
    m = (P == p) & (Q == q)
    if m.sum() < 500: continue
    meas.append(win[m].mean()); prd.append(r); shr.append((ratio[m] > 1 - r).mean())
meas, prd, shr = map(np.array, (meas, prd, shr))
print(f"{len(meas)} position pairs; measured third-win share {meas.min():.2f}-{meas.max():.2f} (mean {meas.mean():.2f})")
print(f"correlation of predicted lean with measured share: {np.corrcoef(prd, meas)[0,1]:.2f}; rank {np.corrcoef(np.argsort(np.argsort(prd)), np.argsort(np.argsort(meas)))[0,1]:.2f}")
print(f"predicted share from the lean alone: mean {shr.mean():.2f}, correlation with measured {np.corrcoef(shr, meas)[0,1]:.2f}, mean abs error {np.abs(shr-meas).mean():.2f}")
for p in (0, 4, 8, 9):
    row = [f"{q}:{meas[i]:.2f}/{shr[i]:.2f}" for i, (pp, q) in enumerate([k for k in pred if ((P == k[0]) & (Q == k[1])).sum() >= 500]) if pp == p]
    print(f"   second at {p}, third at q: measured/predicted " + " ".join(row))

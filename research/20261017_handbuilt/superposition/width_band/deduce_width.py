"""Can the width of the uncertain zone be computed from the weights? Per arrangement, on a canonical near-tie list
(largest 1.539, second 1.0, third 0.99, others at their conditional mean), the exact linear map gives the lead's
coefficients. Model: lead = c_S x_S + c_T x_T + c_L x_L + sum_j c_j x_j, with the seven smaller numbers independent
normals truncated below the third. Then the third wins when the lead < 0; with the smaller numbers as Gaussian noise
(exact truncated-normal mean and variance), P(third | gap, arrangement) = Phi(-mean lead / sd). Predicted curve
against the observed one (curve.log), and the predicted share of lists where the third wins against the true one.
Cost: 720 network runs plus their linear maps; the inputs' distribution is used, no network run on random lists."""
import sys; sys.path.insert(0, "../steer"); sys.path.insert(0, "../level2/deduce")
import numpy as np
from scipy.stats import norm
from harness import *
from case_maps import A, idx, C
C2 = C.clone(); r0 = 0.99; mu_r = -norm.pdf(r0) / norm.cdf(r0)
for i, (l, p, q) in enumerate(A): C2[i] = mu_r; C2[i, l], C2[i, p], C2[i, q] = 1.539, 1.0, 0.99
J = jac(ARC, C2).cpu().numpy()
X = lists(4_000_000, seed=502); o = X.argsort(1, descending=True); v = X.sort(1, descending=True).values
m = v[:, 1] > 0.1; Xn = X[m].cpu().numpy(); on = o[m].cpu().numpy(); vn = v[m].cpu().numpy()
a_true = (run(ARC, X[m]).argmax(1) == o[m][:, 2]).cpu().numpy()
ai = np.array([idx[(l, s, t)] for l, s, t in on[:, :3]])
pred = np.zeros(len(Xn))
for k in np.unique(ai):
    sel = np.where(ai == k)[0]; l, s, t = A[k]; c = J[k, s] - J[k, t]
    others = [j for j in range(10) if j not in (l, s, t)]
    x3 = vn[sel, 2]; alpha = x3                                 # truncation point (standard normal below x3)
    mtr = -norm.pdf(alpha) / norm.cdf(alpha); vtr = 1 - alpha * norm.pdf(alpha) / norm.cdf(alpha) - mtr ** 2
    mean = c[s] * Xn[sel, s] + c[t] * Xn[sel, t] + c[l] * Xn[sel, l] + c[others].sum() * mtr
    sd = np.sqrt((c[others] ** 2).sum() * np.maximum(vtr, 1e-9))
    pred[sel] = norm.cdf(-mean / sd)
g = (vn[:, 1] - vn[:, 2]) / vn[:, 1]
bins = [0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.2, 2]
print("third wins, observed / predicted from the weights, by gap (second-third)/second:")
print("   " + " ".join(f"{lo:.3f}-{hi:.3f}: {100*a_true[(g >= lo) & (g < hi)].mean():.1f}/{100*pred[(g >= lo) & (g < hi)].mean():.1f}" for lo, hi in zip(bins, bins[1:])))
print(f"share of all these lists where the third wins: observed {100*a_true.mean():.2f}%, predicted {100*pred.mean():.2f}%")

"""Diagnosis of the too-wide tail (deduce_width3.log): in the 3-12% gap range, compare the predicted lead (mean and
spread, per arrangement) with the real lead on sampled lists (diagnostic only), in units of the second. Is the mean
too low or the spread too wide? Also split the error of the predicted third-win share by largest/second."""
import numpy as np, torch
from deduce_width3 import *
sc_all = scores(ARC, X[m]); n = torch.arange(len(Xn), device=DEV)
lead_true = ((sc_all[n, o[m][:, 1]] - sc_all[n, o[m][:, 2]]) / v[m][:, 1]).cpu().numpy()
mean_p = np.zeros(len(Xn)); sd_p = np.zeros(len(Xn))
for k in np.unique(ai):
    l, s, t = A[k]
    for gk in range(len(R)):
        sel = np.where((ai == k) & (gi == gk))[0]
        if not len(sel): continue
        sc = vn[sel, 1]
        dtop = Jt[gk, k][l] * (Xn[sel, l] / sc - 1.539) + Jt[gk, k][s] * (Xn[sel, s] / sc - 1.0) + Jt[gk, k][t] * (Xn[sel, t] / sc - R[gk])
        mean_p[sel] = lead0[gk, k] + shift[gk, k] + dtop; sd_p[sel] = np.sqrt(max(var[gk, k], 1e-12))
z = (g >= 0.03) & (g < 0.12)
d = lead_true[z] - mean_p[z]
print(f"gap 3-12%: real lead median {np.median(lead_true[z]):.3f}, predicted mean median {np.median(mean_p[z]):.3f}; real minus predicted mean: median {np.median(d):+.3f}; spread of (real - predicted mean) {np.std(d):.3f} against predicted spread median {np.median(sd_p[z]):.3f}")
q = vn[:, 0] / vn[:, 1]
for lo, hi in ((1, 1.2), (1.2, 1.5), (1.5, 2), (2, 3), (3, 100)):
    k = (q >= lo) & (q < hi)
    print(f"   largest/second {lo}-{hi}: third wins observed {100*a_true[k].mean():.2f}%, predicted {100*pred[k].mean():.2f}%")

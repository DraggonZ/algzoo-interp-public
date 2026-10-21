"""(Near ties are integrated properly in near_ties.py; the main grid does not resolve that region's edge.)
Predicted mistake statistics for ARC (no sampling) against the sampled truth (truth_stats.py, 20M lists):
what wrong answers name, accuracy by arrival order of the three largest, and near ties (second minus third < 5% of
|second|, second > 0) where the third wins."""
import numpy as np
from lib import *
from estimate import estimate
W = load_net("ARC"); z = np.load("truth_stats.npz"); c, N, oc, nt = z["c"], z["N"], z["oc"], z["nt"]
names = ["L,2,3", "L,3,2", "2,L,3", "3,L,2", "2,3,L", "3,2,L"]
rows = {"truth (sampled)": (c[0] / N, c[1:] / (N - c[0]), oc[:, 0] / oc[:, 1], nt[:, 0].sum() / nt[:, 1].sum(), nt[:, 0] / nt[:, 1])}
for label, model, n, q, kind in [("old reduction 0/1", "mean01", 32, 0, "split-gl"), ("two leads, Gaussian", "two", 24, 0, "split-gl"),
                                 ("final (frozen, n=16)", "final", 16, 4, "split3-gl"), ("final, finer grid n=24", "final", 24, 4, "split3-gl"), ("v2 (revised, n=16)", "v2", 16, 4, "split3-gl")]:
    r = estimate(W, n, n, model, q, detail=True, kind=kind)
    print(f"{label}: third > 0: accuracy {100*r['sign_acc'][0]:.2f}% (wrong naming another {100*r['sign_split'][0,2]:.2f} points); third < 0 ({100*r['sign_share'][1]:.1f}% of lists): accuracy {100*r['sign_acc'][1]:.2f}% (wrong naming another {100*r['sign_split'][1,2]:.2f} points)")
    rows[label] = (r["acc"], r["split"] / (1 - r["acc"]), r["order_acc"], r["near_third_all"], r["near_third"])
print("truth (other_wins.log): third > 0 accuracy 96.04%, 'another' 0.22 points; third < 0 (5.5% of lists) accuracy 82.93%, 'another' 10.06 points")
print("arrival order: L = largest, 2 = second, 3 = third, in the order they arrive")
for k, (a, sp, oa, nta, ntp) in rows.items():
    print(f"{k:30s} accuracy {100*a:.2f}%  mistakes naming largest {100*sp[0]:.1f}%, third {100*sp[1]:.1f}%, another {100*sp[2]:.1f}%")
    print(" " * 31 + "by arrival order, accuracy: " + "  ".join(f"{names[i]} {100*oa[i]:.2f}" for i in range(6)))

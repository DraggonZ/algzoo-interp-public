"""Can the half-degree tolerance for the cancellers' direction be predicted from the readout? Turning the four
cancellers' final state h_C by angle theta changes the lead by at most theta * |d_C| * |h_C|, where d_C is the readout
difference (winner minus runner-up) restricted to the four. Predicted tolerance per list: lead / (|d_C| |h_C|).
Compare with direction_vs_size.log (0.5 deg: 93%, 1 deg: 87%, 5 deg: 47%)."""
import numpy as np
import simlib

X = simlib.lists(7, 100_000); L, _, H = simlib.run(X); C = [1, 6, 7, 8]
o = np.argsort(L, 1); w, r = o[:, -1], o[:, -2]; lead = L[np.arange(len(X)), w] - L[np.arange(len(X)), r]
d = simlib.WO[w][:, C] - simlib.WO[r][:, C]; hC = H[:, -1, C]
tol = np.degrees(lead / (np.linalg.norm(d, axis=1) * np.linalg.norm(hC, axis=1) + 1e-12))
q = np.quantile(tol, [.1, .25, .5, .75])
print("predicted tolerance (deg) quantiles 10/25/50/75%:", np.round(q, 2))
for th in (0.5, 1, 5):
    print(f"share of lists whose tolerance is below {th} deg: {np.mean(tol < th):.3f}")
print("share of the four's push on the lead in the plane of worst turning, median |d_C||h_C| / lead:", round(float(np.median(1 / np.radians(tol))), 1))

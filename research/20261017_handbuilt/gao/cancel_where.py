"""Where does the cancellation happen? Pushes on the winner's lead, (w_o[winner] - w_o[runner-up])_n h_n, summed by
neuron group: per group, the median absolute net push and its share of the total absolute push; and how much of the
total cancels within groups versus between groups. Plus one worked example list. ARC's network, 100,000 lists."""
import numpy as np
import simlib

G = {"timer/memory (0,3,10-15)": [0, 3, 10, 11, 12, 13, 14, 15], "cancelling (1,6,7,8)": [1, 6, 7, 8],
     "store+detector (2,4)": [2, 4], "gates (5,9)": [5, 9]}
X = simlib.lists(2323, 100_000); r = np.arange(len(X))
L, _, H = simlib.run(X); k = L.argmax(1); j = np.argsort(L, 1)[:, -2]; lead = L[r, k] - L[r, j]
P = (simlib.WO[k] - simlib.WO[j]) * H[:, 9]; tot = np.abs(P).sum(1)
net = {g: P[:, ns].sum(1) for g, ns in G.items()}; within = {g: np.abs(P[:, ns]).sum(1) for g, ns in G.items()}
print(f"median lead {np.median(lead):.1f}; median total |push| {np.median(tot):.0f}")
for g in G:
    print(f"{g:26s}: median net push on the lead {np.median(net[g]):+7.1f} (|net| {np.median(np.abs(net[g])):6.1f}); share of total |push| {np.median(within[g] / tot):.2f}")
between = np.abs(np.stack(list(net.values()), 1)).sum(1)
print(f"cancelling within groups: median (sum of |push| - sum of |group net|) / total = {np.median((tot - between) / tot):.2f}; "
      f"between groups: median (sum of |group net| - |lead|) / total = {np.median((between - np.abs(lead)) / tot):.2f}")
i = int(np.argsort(np.abs(np.log(tot / 390)) + np.abs(np.log(np.maximum(lead, 1e-6) / 8)))[0])
print("worked example list:", np.round(X[i], 2), "answer", k[i], "runner-up", j[i], "lead", round(float(lead[i]), 1))
print("   pushes on the lead by neuron:", " ".join(f"{n}:{P[i, n]:+.0f}" for n in range(16) if abs(P[i, n]) > 0.5))
print("   by group:", "; ".join(f"{g.split(' ')[0]} {net[g][i]:+.0f}" for g in G), f"; total |push| {tot[i]:.0f}")

# Is the within-group cancellation a matter of basis? Rotate each group's final activations onto its principal
# directions (centred PCA over lists) and recompute the pushes in that basis: the lead is unchanged, the total |push|
# is not. Also a "best" basis per group: the readout-difference direction itself (one push per group, no cancellation
# within it), which is the lower bound.
tot_pca = np.zeros(len(X)); tot_min = np.zeros(len(X))
for g, ns in G.items():
    Hg = H[:, 9][:, ns]; mu = Hg.mean(0); U, S, Vt = np.linalg.svd(Hg - mu, full_matrices=False)
    W = (simlib.WO[k] - simlib.WO[j])[:, ns]
    tot_pca += np.abs((W @ Vt.T) * ((Hg - mu) @ Vt.T)).sum(1) + np.abs((W * mu).sum(1))
    tot_min += np.abs((W * Hg).sum(1))
print(f"total |push| in the neuron basis {np.median(tot):.0f}; in each group's principal directions {np.median(tot_pca):.0f}; "
      f"one push per group {np.median(tot_min):.0f}. Condition number (total / lead): {np.median(tot / lead):.0f}, {np.median(tot_pca / lead):.0f}, {np.median(tot_min / lead):.0f}")

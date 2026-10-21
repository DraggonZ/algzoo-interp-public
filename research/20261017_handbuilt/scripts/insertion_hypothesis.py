"""Hypothesis: at a second-third tie (both after the largest), the winner is set by
neuron 5's strength at each insertion (n5 one step after arrival) times an age gain.
Reduced inputs: largest 1.6 at p1, second 1.0 at p2, third 1.0*(1-1e-6) at p3, others -0.46.
Margin M = s[p2] - s[p3] (centred). Compare with n5 strengths and ages."""
import itertools

import numpy as np
import torch as th

from algzoo.core import example, forward

P = example()
rows = []
for p1, p2, p3 in itertools.permutations(range(10), 3):
    if not (p1 < p2 and p1 < p3) or max(p2, p3) == 9:
        continue
    x = np.full(10, -0.46); x[p1] = 1.6; x[p2] = 1.0; x[p3] = 1.0 - 1e-6
    s, H = forward(P, th.tensor(x[None]), return_hidden=True)
    s = s[0].numpy(); H = H[0].numpy()
    n5_2, n5_3 = H[p2 + 2, 5], H[p3 + 2, 5]  # H index t+1 holds state after input t; n5 one step after arrival
    rows.append((p1, p2, p3, s[p2] - s[p3], n5_2, n5_3))
r = np.array(rows)
M, a2, a3 = r[:, 3], r[:, 4], r[:, 5]
print(f"{len(r)} placements; margin sd {M.std():.2f}; third wins at the tie in {np.mean(M < 0):.2f}")
print(f"n5 at insertion: second's {np.median(a2):.1f} (range {a2.min():.1f}-{a2.max():.1f}), third's {np.median(a3):.1f}")
# does the n5 difference predict the margin sign? add age gains via a per-age fit: M ~ G[age2]*a2 - G[age3]*a3
age2, age3 = 9 - r[:, 1].astype(int), 9 - r[:, 2].astype(int)
print("sign(M) == sign(n5 second - n5 third):", np.mean(np.sign(M) == np.sign(a2 - a3)).round(2))
A = np.zeros((len(r), 10))
A[np.arange(len(r)), age2] += a2; A[np.arange(len(r)), age3] -= a3
G, *_ = np.linalg.lstsq(A, M, rcond=None)
pred = A @ G
print("fit M ~ G[age]*n5 (10 numbers): R^2", round(1 - ((pred - M) ** 2).mean() / M.var(), 2), " sign agreement", np.mean(np.sign(pred) == np.sign(M)).round(2))
# cross-validated sign agreement (leave out by p1)
agree = []
for q in range(10):
    tr, te = r[:, 0] != q, r[:, 0] == q
    if te.sum() == 0:
        continue
    Gq, *_ = np.linalg.lstsq(A[tr], M[tr], rcond=None)
    agree.append(np.sign(A[te] @ Gq) == np.sign(M[te]))
print("held out by largest's slot: sign agreement", np.mean(np.concatenate(agree)).round(2))
# how much does n5 strength depend on the largest's age at insertion?
gap2 = (r[:, 1] - r[:, 0]).astype(int)
print("n5 per unit at insertion by steps since the largest arrived:", {int(k): round(float(np.median(a2[gap2 == k])), 1) for k in range(1, 9)})

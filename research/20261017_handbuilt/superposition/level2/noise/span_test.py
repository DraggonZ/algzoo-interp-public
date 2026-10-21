"""Do candidate explanation quantities carry the information that sets a noisy copy's accuracy? Each candidate quantity
here is linear in the weight change (q = a . dw), so a set of them spans a subspace of weight space. Test: keep only the
part of each copy's noise inside that subspace (orthogonal projection; the noise is isotropic, so this is the best guess
of the noise given the quantities) and propagate it to first order through 50,000 lists' nine leads, as in
hilton_lowrank.py. This measures what the quantities could predict with a perfect downstream model, separately from
the question of fitting a small model. Truth: the 200 copies at s = 0.0025, 1M lists each.

Quantities (fixed source weightings from the clean network; only the copy's weights change):
 - level of neuron n at K steps after the largest, per unit of it: dW_hh[n, :] . E[h(t-1) / m1 | K] / c_n
 - the same split into the parts that follow the largest m1 and the runner-up m2 (least-squares of h(t-1) on m1, m2 by K)
"""
from leads import *
print(f"constant guess RMSE {100 * y.std():.2f}; all 432 directions:", rep(propagate(th.eye(P, device=DEV, dtype=DT))))
V = th.as_tensor(np.load("dirs.npz")["V"], device=DEV)
for r in (1, 2, 5, 10, 20): print(f"top {r} sensitive directions ({r} numbers):", rep(propagate(V[:, :r].T)))
cands = {
    "level of each hiding neuron, all steps (4)": [row_vec(nn, allK) for nn in HIDE],
    "level of each hiding neuron by K (4 x 4 = 16)": [row_vec(nn, lvl[b]) for nn in HIDE for b in BINS],
    "level of each hiding neuron, K = 2-6 (4)": [row_vec(nn, (lvl["K=2-3"] + lvl["K=4-6"]) / 2) for nn in HIDE],
    "hiding: parts following the largest and the runner-up, by K (4 x 4 x 2 = 32)": [row_vec(nn, split[b][i]) for nn in HIDE for b in BINS for i in (0, 1)],
    "hiding: parts following the largest and the runner-up, K = 2-6 (8)": [row_vec(nn, (split["K=2-3"][i] + split["K=4-6"][i]) / 2) for nn in HIDE for i in (0, 1)],
    "hiding + memory + store levels, all steps (13)": [row_vec(nn, allK) for nn in HIDE + MEM + [2]],
    "hiding + memory + store levels by K (52)": [row_vec(nn, lvl[b]) for nn in HIDE + MEM + [2] for b in BINS],
    "all 16 neurons, largest and runner-up parts by K (128)": [row_vec(nn, split[b][i]) for nn in range(16) for b in BINS for i in (0, 1)],
    "all recurrent weights into the hiding neurons (64)": [row_vec(nn, th.eye(16, device=DEV, dtype=DT)[j]) for nn in HIDE for j in range(16)],
}
for k, A in cands.items(): print(f"{k}: {rep(propagate(th.stack(A)))}")

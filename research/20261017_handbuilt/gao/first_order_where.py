"""Where do the costly errors of the all-16 menu formulas come from? First-order lead change per (neuron, step),
sum_t g e (first_order_subst.py). On lists the prediction flips, share of the harmful change (negative part summed)
by neuron group and by step. 50,000 lists."""
import io, contextlib
import numpy as np
with contextlib.redirect_stdout(io.StringIO()):
    from block_recurrent import Xs, ys, plain, G
import simlib

X, y = Xs, ys; r = np.arange(len(X))
L, P, H = simlib.run(X); k = L.argmax(1); j = np.argsort(L, 1)[:, -2]; lead = L[r, k] - L[r, j]
g = simlib.WO[k] - simlib.WO[j]; Gt = np.zeros_like(H)
for t in range(9, -1, -1):
    Gt[:, t] = g; g = (g * (P[:, t] > 0)) @ simlib.WH
F = plain(list(range(16))); E = np.stack([F[n] for n in range(16)], -1) - H
C = Gt * E; flip = (lead + C.sum((1, 2)) < 0) & (k == y)
print(f"lists the first-order prediction flips from right to wrong: {flip.mean():.3f}")
harm = np.minimum(C[flip], 0); tot = harm.sum()
for gname, ns in G.items():
    print(f"  {gname:15s}: share of the harmful change {harm[:, :, ns].sum() / tot:.2f}")
print("  by step: " + " ".join(f"{t}:{harm[:, t].sum() / tot:.2f}" for t in range(10)))
print("  by neuron: " + " ".join(f"{n}:{harm[:, :, n].sum() / tot:.2f}" for n in range(16)))
# same with the block account's cancellers? compare: error of the formulas on active vs inactive steps
act = H > 0
print(f"  share of harm on steps where the neuron is truly on: {harm[act[flip]].sum() / tot:.2f}; off (formula fires wrongly): {harm[~act[flip]].sum() / tot:.2f}")

"""Span test (see span_test.py) for (a) the closed-form structural quantities of quantities.py (gradients by central
differences), and (b) greedy selection among the 52 'level per unit of the largest by age' quantities (13 neurons:
hiding, memory, store; 4 age bins): which few carry the noise response?"""
from leads import *
from quantities import quantities, ARC_ROLES

def rep(p): return f"RMSE {100 * np.sqrt(np.mean((p - y) ** 2)):.2f} points, correlation {np.corrcoef(p, y)[0, 1]:.3f}"
eps = 1e-6; I = th.eye(P, device=DEV, dtype=DT)
qp, qm = quantities(W, eps * I), quantities(W, -eps * I)
closed = {k: th.as_tensor((qp[k] - qm[k]) / (2 * eps), device=DEV) for k in qp}
groups = {
    "hiding block eigenvalues (3)": [k for k in closed if k.startswith("hiding:")],
    "comparator gains (4)": [k for k in closed if k.endswith(": gain")],
    "store and detector ratios (4)": [k for k in closed if k.startswith("store:") or k.startswith("detector")],
    "store's share of each reference (4)": [k for k in closed if "store's share" in k],
    "memory eigenvalues (8 distinct)": [k for k in closed if k.startswith("memory: eigenvalue")],
    "late gate weights (6)": [k for k in closed if k.startswith("late gate")],
}
allc = []
for gname, ks in groups.items():
    A = th.stack([closed[k] for k in ks]); A = A[A.norm(dim=1) > 1e-9]; allc.append(A)
    print(f"closed form, {gname}: {rep(propagate(A))}")
print(f"closed form, all of the above together ({sum(len(a) for a in allc)}): {rep(propagate(th.cat(allc)))}")

names, vecs = [], []
for nn in HIDE + MEM + [2]:
    for b in BINS:
        names.append(f"neuron {nn} level, {b}"); vecs.append(row_vec(nn, lvl[b]))
chosen = []
print("\ngreedy selection among the 52 levels by age:")
for step in range(16):
    best = None
    for i in range(len(names)):
        if i in chosen: continue
        p = propagate(th.stack([vecs[j] for j in chosen + [i]])); e = np.sqrt(np.mean((p - y) ** 2))
        if best is None or e < best[0]: best = (e, i, p)
    chosen.append(best[1]); print(f"  {len(chosen):2d}: + {names[best[1]]:28s} {rep(best[2])}")
np.save("greedy_levels.npy", np.array(chosen))

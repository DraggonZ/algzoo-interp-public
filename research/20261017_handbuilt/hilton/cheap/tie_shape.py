"""Development look: does the tie margin D (score of the true second minus the best other, at the tie) of each triple
follow a shared shape in u and in the other gap? For the 2nd-3rd tie (m2 = m3 = u, m1 = u + b) and the 1st-2nd tie
(m1 = m2 = u + a, m3 = u): D on a grid of (u, other gap) for all 720 triples; fit D(u, g) = r(u, g) * D(node) across
triples and report how much of D's spread that leaves."""
import sys, numpy as np, torch
import common as c
log = c.Log("tie_shape.log")
name = sys.argv[1] if len(sys.argv) > 1 else "ARC"
W = dict(c.nets())[name]; P = c.T; n = 720
us = np.array([0.12, 0.3, 0.48, 0.66, 0.84, 1.02, 1.2, 1.5, 2.0]); gs = np.array([0.05, 0.15, 0.3, 0.47, 0.7, 1.0, 1.5])
for tie in ("23", "12"):
    D = np.zeros((n, len(us), len(gs)))
    for i, u in enumerate(us):
        for j, gg in enumerate(gs):
            v = [u + gg, u, u] if tie == "23" else [u + gg, u + gg, u]
            S, _ = c.scores(W, c.inputs(P, torch.tensor([v], device=c.DEV, dtype=c.dt).repeat(n, 1)))
            D[:, i, j] = c.margin_from(S, P[:, 1])[0].cpu().numpy()
    i0, j0 = 4, 3 if tie == "23" else 2
    los = D[:, i0, j0] < 0
    log(f"== tie {tie}, net {name}: node u={us[i0]}, gap={gs[j0]}; wrong at the node's tie: {los.sum()}; share of all (triple, u, gap) wrong at the tie that are wrong at the node: {((D<0)&los[:,None,None]).sum()/(D<0).sum():.3f}")
    log("  rows u, cols gap " + " ".join(f"{x:5.2f}" for x in gs) + ": mean D over triples wrong at the node, then R^2 of D = r * D(node)")
    for i, u in enumerate(us):
        row = []; r2 = []
        for j in range(len(gs)):
            x = D[los, i0, j0]; y = D[los, i, j]; r = (x * y).sum() / (x * x).sum(); r2.append(1 - ((y - r * x) ** 2).sum() / ((y - y.mean()) ** 2).sum())
            row.append(y.mean())
        log(f"  u={u:4.2f} " + " ".join(f"{x:6.2f}" for x in row) + "  | R2 " + " ".join(f"{x:5.2f}" for x in r2))
    np.save(f"tie_shape_{tie}_{name}.npy", D)

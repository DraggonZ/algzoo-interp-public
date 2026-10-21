"""Question: is the overtaken record's tail (its wave at positions after the overtaking) cut by memory switching?
Test: run the trained network with chosen memory neurons left unclipped (identity instead of ReLU), and repeat the
double nudge of tail_cancel2.py (record at r and overtaking number at q nudged together). Report the total score change
summed over positions after q, and accuracy, for: no change; neuron 0 unclipped; neuron 15 unclipped; all memory
neurons unclipped."""
import numpy as np
from algzoo.core import example
wi, wh, wo = (np.asarray(p, float) for p in example())
rng = np.random.default_rng(3); X = rng.standard_normal((1500000, 10)); y = np.argsort(X, 1)[:, -2]
def final(X, lin):
    h = np.zeros((len(X), 16)); m = np.zeros(16, bool); m[lin] = True
    for t in range(10):
        pre = h @ wh.T + X[:, t:t + 1] * wi; h = np.where(m, pre, np.maximum(pre, 0))
    return h
eps = 1e-3
for name, lin in [("trained as is", []), ("neuron 0 unclipped", [0]), ("neuron 15 unclipped", [15]), ("all memory unclipped", [0, 3, 10, 11, 12, 13, 14, 15])]:
    acc = ((final(X[:200000], lin) @ wo.T).argmax(1) == y[:200000]).mean()
    out = []
    for r, q in [(1, 4), (2, 5), (2, 7), (4, 7)]:
        pre = np.maximum(X[:, :r].max(1), 0)
        ok = (X[:, r] > pre + 0.05) & (X[:, r] > 0.3) & (X[:, r + 1:q].max(1) < X[:, r]) & (X[:, q] > X[:, r] + 0.05)
        if q < 9: ok &= X[:, q + 1:].max(1) < X[:, q]
        Xs = X[ok]; Xp = Xs.copy(); Xp[:, r] += eps; Xp[:, q] += eps
        tot = np.median((final(Xp, lin) - final(Xs, lin)) @ wo.T / eps, 0)
        out.append(f"r{r} q{q}: own {tot[r]:6.1f}, after q {tot[q + 1:9].sum():7.1f}")
    print(f"{name:22s} accuracy {acc:.3f} | " + " | ".join(out))

"""Falsification test of 'the rest of the error is free': add random errors to all 16 neurons
at every step (scale s x each neuron's typical active value, on active steps), then (a) as drawn, (b) projected
orthogonal to the list's backward direction g (all steps and neurons; computed from the true run), rescaled to the same
size, (c) along g only, same size, random sign per list. Also the first-order predictions. Substitute (clipped at zero) and measure accuracy. 50,000 lists."""
import numpy as np
import simlib

X = simlib.lists(5151, 50_000); y = X.argsort(1)[:, -2]; r = np.arange(len(X))
L, P, H = simlib.run(X); k = L.argmax(1); j = np.argsort(L, 1)[:, -2]
g = simlib.WO[k] - simlib.WO[j]; Gt = np.zeros_like(H)
for t in range(9, -1, -1):
    Gt[:, t] = g; g = (g * (P[:, t] > 0)) @ simlib.WH
typ = np.array([np.median(H[:, :, n][H[:, :, n] > 0]) for n in range(16)]); act = H > 0
Gm = Gt * act; gn = np.sqrt((Gm ** 2).sum((1, 2), keepdims=True)) + 1e-12; u = Gm / gn
rng = np.random.default_rng(0); acc0 = (k == y).mean()
print(f"network {acc0:.3f}")
for s in (0.01, 0.02, 0.05, 0.1):
    E = rng.standard_normal(H.shape) * typ * s * act; size = np.sqrt((E ** 2).sum((1, 2), keepdims=True))
    along = (E * u).sum((1, 2), keepdims=True) * u; orth = E - along
    orth *= size / (np.sqrt((orth ** 2).sum((1, 2), keepdims=True)) + 1e-12)
    par = u * size * rng.choice([-1.0, 1.0], size=(len(X), 1, 1))   # random sign per list
    out = []
    for lab, e in (("as drawn", E), ("orthogonal to g", orth), ("along g", par)):
        sims = {n: np.maximum(H[:, :, n] + e[:, :, n], 0) for n in range(16)}
        out.append(f"{lab} {(simlib.run(X, sims)[0].argmax(1) == y).mean():.3f}")
    lead = L[r, k] - L[r, j]
    fo = {lab: np.where(lead + (Gt * e).sum((1, 2)) < 0, j == y, k == y).mean() for lab, e in (("as drawn", E), ("orthogonal", orth), ("along g", par))}
    print(f"errors {s:.0%} of typical values: " + "; ".join(out) + " | first-order predicted: " + "; ".join(f"{a} {b:.3f}" for a, b in fo.items()), flush=True)

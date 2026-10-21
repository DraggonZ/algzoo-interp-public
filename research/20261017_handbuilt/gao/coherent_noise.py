"""Are independent per-neuron errors what the cancellation amplifies? Noise of relative size s on the four cancelling
neurons (1, 6, 7, 8) or the memory neurons, at every step: (a) independent, each neuron's value times (1 + s z_n);
(b) shared, all neurons of the group times the same (1 + s z) per list and step, which keeps the ratios inside the
group. Same relative error per neuron in both. ARC's network, 100,000 lists."""
import numpy as np
import simlib

X = simlib.lists(2424, 100_000); y = X.argsort(1)[:, -2]; B = len(X); acc0 = (simlib.run(X)[0].argmax(1) == y).mean()
rng = np.random.default_rng(0)
G = {"cancelling (1,6,7,8)": [1, 6, 7, 8], "memory (0,3,10-15)": [0, 3, 10, 11, 12, 13, 14, 15], "all 16": list(range(16))}
print(f"accuracy {acc0:.3f}")
for g, ns in G.items():
    for s in (0.01, 0.03, 0.1):
        out = []
        for shared in (False, True):
            h = np.zeros((B, 16))
            for t in range(10):
                h = np.maximum(h @ simlib.WH.T + X[:, t:t + 1] * simlib.WI, 0)
                z = rng.standard_normal((B, 1)) if shared else rng.standard_normal((B, len(ns)))
                h[:, ns] *= 1 + s * z
                h = np.maximum(h, 0)
            out.append(((h @ simlib.WO.T).argmax(1) == y).mean())
        print(f"{g:22s} relative noise {s:.2f}: independent per neuron {out[0]:.3f}, shared by the group {out[1]:.3f}", flush=True)

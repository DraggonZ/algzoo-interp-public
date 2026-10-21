"""Readout cancellation or compounding through steps? Errors of relative size e on the neurons' values applied
(a) only to the final state that the readout reads, or (b) at every step (fed through the recurrence). Shared scaling
(all neurons x 1.02) and independent per-neuron noise (each value x (1 + e z)). ARC's network, 100,000 lists."""
import numpy as np
import simlib

X = simlib.lists(3232, 100_000); y = X.argsort(1)[:, -2]; B = len(X); rng = np.random.default_rng(0)
acc0 = (simlib.run(X)[0].argmax(1) == y).mean(); print(f"accuracy {acc0:.3f}")
for kind in ("shared scale", "independent noise"):
    for e in (0.01, 0.02, 0.05, 0.1):
        res = []
        for where in ("readout only", "every step"):
            h = np.zeros((B, 16))
            for t in range(10):
                h = np.maximum(h @ simlib.WH.T + X[:, t:t + 1] * simlib.WI, 0)
                if where == "every step" or t == 9:
                    f = (1 + e) if kind == "shared scale" else (1 + e * rng.standard_normal(h.shape))
                    h = np.maximum(h * f, 0)
            res.append(((h @ simlib.WO.T).argmax(1) == y).mean())
        print(f"{kind:17s} {e:.2f}: readout only {res[0]:.3f}; every step {res[1]:.3f}", flush=True)

"""How often memory neurons switch off (pre-activation below zero while the slot is in use) as the spread memory's level
is lowered; accuracy alongside."""
import numpy as np
from algzoo.handbuilt_v23 import build_v23, spread_memory, scores
def pre_acts(net, X):
    wi, wh, _ = net.params; h = np.zeros((len(X), len(wi))); Z = []
    for t in range(X.shape[1]):
        z = h @ wh.T + X[:, t:t + 1] * wi; Z.append(z); h = np.maximum(z, 0)
    return np.stack(Z, 1)
if __name__ == "__main__":
    rng = np.random.default_rng(5); X = rng.standard_normal((100000, 10)); S = np.argsort(-X, 1)[:, 1]
    base = build_v23(); Yi = [base.idx[f"mem{j}"] for j in range(8)]; U = 4.0
    for lam in (0, -1, -2, -3, -3.5, -4):
        n = spread_memory(base, lam); Z = pre_acts(n, X)[:, :, Yi]
        print(f"extra level {lam:+.1f} (shared level {U + lam:.1f} x m1): accuracy {(scores(n, X).argmax(1) == S).mean():.4f}, memory neuron-steps switched off {(Z < -1e-9).mean():.3f}, lists with any {((Z < -1e-9).any((1, 2))).mean():.2f}")

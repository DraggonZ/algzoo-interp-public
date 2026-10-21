"""How faithful is the matched hand-built network to the trained one, measured with the trained network's own weights?
No fitting: the hand-built network (v18 at gains 10 and with clock gain 50) is relabelled and rescaled to the trained
coordinates by exact symmetries (match_order_scale2.py; same function). On random lists:
(a) Answer: the trained readout applied to the hand-built network's final state (matched), against the true second-largest.
(b) One step: for each step, the trained network's update applied to the hand-built state, relu(W h* + w x), against
    the hand-built network's own next state h*; variance of h* matched, per neuron group (this is the 'discrepancy noise'
    that a Hilton-style propagation would have to carry).
(c) Trained network's state against the hand-built state at the same step: variance of the trained neuron matched by
    the hand-built neuron in its slot (no regression; same coordinates)."""
import numpy as np
from algzoo.core import example
T = [np.asarray(p, float) for p in example()]
rng = np.random.default_rng(31); X = rng.standard_normal((200000, 10)); y = np.argsort(X, 1)[:, -2]
G = {"largest pair (2, 4)": [2, 4], "gates (5, 9)": [5, 9], "hiding (1, 6, 7, 8)": [1, 6, 7, 8], "memory (0, 3, 10-15)": [0, 3, 10, 11, 12, 13, 14, 15]}
def states(P, X):
    wi, wh, wo = P; h = np.zeros((len(X), len(wi))); hs = [h]
    for t in range(10): h = np.maximum(h @ wh.T + X[:, t:t + 1] * wi, 0); hs.append(h)
    return np.stack(hs, 1)
HT = states(T, X)
for f in ("v18ts_matched2.npz", "v18g50_matched.npz"):
    d = np.load(f); H = (d["wi"], d["wh"], d["wo"]); HS = states(H, X)
    own = ((HS[:, -1] @ H[2].T).argmax(1) == y).mean(); plug = ((HS[:, -1] @ T[2].T).argmax(1) == y).mean()
    print(f"== {f}: hand-built's own answer {own:.3f}; trained readout on the hand-built final state {plug:.3f}")
    nxt = np.maximum(np.einsum("btj,ij->bti", HS[:, :-1], T[1]) + X[:, :, None] * T[0][None, None, :], 0)
    for g, idx in G.items():
        a = nxt[:, 3:, idx].reshape(-1, len(idx)); b = HS[:, 4:, idx].reshape(-1, len(idx))
        r2 = 1 - ((a - b) ** 2).sum(0) / ((b - b.mean(0)) ** 2).sum(0)
        c = HT[:, 4:, idx].reshape(-1, len(idx)); r2c = 1 - ((c - b) ** 2).sum(0) / ((c - c.mean(0)) ** 2).sum(0)
        print(f"   {g:22s} (b) trained update on hand-built state, per neuron: " + " ".join(f"{v:6.2f}" for v in r2)
              + f" | (c) trained vs hand-built value: " + " ".join(f"{v:6.2f}" for v in r2c))

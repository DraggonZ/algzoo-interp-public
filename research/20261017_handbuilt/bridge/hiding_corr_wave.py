"""Review's cheap check: the hiding neurons of v18 (clock 0-2 and flip, trained gain scale) matched trained 7, 6, 8 and
1 at activity correlations 0.90-1.00 (v18ts_match2.log). Recompute with each group's mean over its four neurons removed
at every step, so the shared level does not count. Steps 3-9, 200,000 lists. Pairing as in match_order_scale2.py."""
import numpy as np
from algzoo.core import example
from algzoo.handbuilt_v18 import build_v18
rng = np.random.default_rng(5); X = rng.standard_normal((200000, 10))
def states(P, X):
    wi, wh, wo = (np.asarray(p, float) for p in P); h = np.zeros((len(X), len(wi))); hs = []
    for t in range(10): h = np.maximum(h @ wh.T + X[:, t:t + 1] * wi, 0); hs.append(h)
    return np.stack(hs, 1)
HT = states(example(), X); net = build_v18(g=10, a=10, w=10, lam=10.0, c=-9.0, eps=0.1); I = net.idx; H = states(net.params, X)
pairs = [("clock0", 7), ("clock1", 6), ("clock2", 8), ("flip", 1)]
hv = H[:, 3:, [I[k] for k, _ in pairs]].reshape(-1, 4); tv = HT[:, 3:, [n for _, n in pairs]].reshape(-1, 4)
hw = hv - hv.mean(1, keepdims=True); tw = tv - tv.mean(1, keepdims=True)
for i, (k, n) in enumerate(pairs):
    print(f"{k:7s} vs trained {n}: correlation {np.corrcoef(hv[:, i], tv[:, i])[0, 1]:.2f}; with group means removed {np.corrcoef(hw[:, i], tw[:, i])[0, 1]:.2f}")
print(f"group means: correlation {np.corrcoef(hv.mean(1), tv.mean(1))[0, 1]:.2f}; share of the trained group's variance in its mean {tv.mean(1).var() * 4 / tv.var(0).sum():.2f}")

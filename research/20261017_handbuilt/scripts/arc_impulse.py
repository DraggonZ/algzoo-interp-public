"""Write patterns of the trained network. With each random list's on/off pattern fixed, the final scores respond
linearly to an extra unit of activity in one neuron at step t. Average that response (scores minus their mean) over
lists, for the detector (4), the store (2) and the late gate (9), by step t. Lists: at least two positive numbers."""
import numpy as np
from algzoo.core import example
wi, wh, wo = (np.asarray(w, dtype=float) for w in example()); wi = wi.reshape(-1)
X = np.random.default_rng(43).standard_normal((100_000, 10)); X = X[(X > 0).sum(1) >= 2]; n = len(X)
h = np.zeros((n, 16)); ons = []
for t in range(10):
    pre = h @ wh.T + X[:, t:t + 1] * wi; ons.append((pre > 0).astype(float)); h = np.maximum(pre, 0)
lam = [None] * 10; lam[9] = np.broadcast_to(wo - wo.mean(0), (n, 10, 16))
for t in range(8, -1, -1): lam[t] = np.einsum("npj,jk->npk", lam[t + 1] * ons[t + 1][:, None, :], wh)
np.set_printoptions(precision=1, suppress=True, linewidth=170)
for k, name in ((4, "detector"), (2, "store"), (9, "late gate")):
    print(f"--- one unit of {name} at step t: change in each final score (positions 0-9), mean over lists")
    for t in range(10):
        print(f"  t={t}: " + " ".join(f"{v:6.1f}" for v in lam[t][:, :, k].mean(0)))

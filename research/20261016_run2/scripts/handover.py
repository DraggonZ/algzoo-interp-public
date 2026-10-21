"""Which neurons carry the 2nd spike's age through the max's arrival? (deterministic θ grid, base model)

Case B: the 2nd largest arrives first (p2 < p1). Just before the max arrives (after step p1 - 1), zero either the
6 comparator neurons or the 10 dense neurons, then continue. Compare two-spike accuracy with the unablated network.
Case A (max first) is shown with the same ablation just before the 2nd spike, as a contrast.
"""
import numpy as np
import torch as th

from algzoo.core import example
from algzoo.exact_z2 import HI, LO, g_angle

dev = "cuda"
NG = 1000
tg = LO + (np.arange(NG) + 0.5) * (HI - LO) / NG
W = th.as_tensor(g_angle(tg) * (HI - LO) / NG, device=dev)
cs, sn = th.as_tensor(np.cos(tg), device=dev), th.as_tensor(np.sin(tg), device=dev)
C = [1, 2, 4, 6, 7, 8]
Dn = [i for i in range(16) if i not in C]
w_hi, w_hh, w_oh = (th.as_tensor(p, device=dev, dtype=th.float64) for p in example())


@th.no_grad()
def acc(p1, p2, zero=None):
    second = max(p1, p2)
    h = th.zeros(NG, 16, device=dev, dtype=th.float64)
    for t in range(10):
        if t == second and zero is not None:
            h = h.clone()
            h[:, zero] = 0
        x = cs if t == p1 else (sn if t == p2 else th.zeros_like(cs))
        h = th.relu(h @ w_hh.T + x[:, None] * w_hi)
    ok = (h @ w_oh.T).argmax(-1) == p2
    return (W * ok).sum().item() * 90, W.sum().item() * 90


for case, cond in [("B: 2nd first, then max", lambda a, b: b < a), ("A: max first, then 2nd", lambda a, b: a < b)]:
    rows = {"none": [], "comparators zeroed": [], "dense zeroed": []}
    for p1 in range(10):
        for p2 in range(10):
            if p1 == p2 or not cond(p1, p2):
                continue
            for lab, z in [("none", None), ("comparators zeroed", C), ("dense zeroed", Dn)]:
                rows[lab].append(acc(p1, p2, z)[0])
    print(case + ": mean two-spike accuracy over pairs " + ", ".join(f"{k} {np.mean(v):.3f}" for k, v in rows.items()))

"""Event 2 on full random inputs: when a new largest arrives, does the top vote move to the old largest's slot?

Also event 1: when a new second arrives (between the current second and the largest), does the top vote move to the
newest slot? Reported for ARC's model and seeds 0-4, by how much the new input beats the old value.
"""
import numpy as np
import torch as th

from algzoo.core import example, zoo

import sys
dev = "cuda"
T0 = int(sys.argv[1]) if len(sys.argv) > 1 else 2
for name in ["example", 0, 1, 2, 3, 4]:
    p = example() if name == "example" else zoo(16, 10, name)
    w = [th.as_tensor(q, device=dev, dtype=th.float32) for q in p]
    g = th.Generator(device=dev).manual_seed(11)
    X = th.randn(200000, 10, device=dev, generator=g)
    h = th.zeros(X.shape[0], 16, device=dev)
    res = {"new largest -> old largest's slot": [0, 0], "new second -> newest slot": [0, 0]}
    near = {"new largest": [0, 0], "new second": [0, 0]}
    for t in range(10):
        h = th.relu(h @ w[1].T + X[:, t, None] * w[0])
        if t < T0:
            continue
        top = (h @ w[2].T).argmax(-1)
        prev = X[:, :t]
        M, iM = prev.max(-1)
        m = prev.topk(2, -1).values[:, 1]
        x = X[:, t]
        a = x > M
        ok = top[a] == iM[a] + (9 - t)
        res["new largest -> old largest's slot"][0] += ok.sum().item(); res["new largest -> old largest's slot"][1] += a.sum().item()
        b = (x > m) & (x < M)
        ok2 = top[b] == 9
        res["new second -> newest slot"][0] += ok2.sum().item(); res["new second -> newest slot"][1] += b.sum().item()
        na = a & (x - M < 0.1); nb = b & ((x - m < 0.1) | (M - x < 0.1))
        near["new largest"][0] += (top[na] == iM[na] + (9 - t)).sum().item(); near["new largest"][1] += na.sum().item()
        near["new second"][0] += (top[nb] == 9).sum().item(); near["new second"][1] += nb.sum().item()
    print(f"{name}: " + "; ".join(f"{k}: {v[0]/v[1]:.3f}" for k, v in res.items()) +
          f" | within 0.1 of a tie: new largest {near['new largest'][0]/near['new largest'][1]:.3f}, new second {near['new second'][0]/near['new second'][1]:.3f}")

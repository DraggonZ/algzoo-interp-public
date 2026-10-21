"""ARC 'conveyor belt': p_t = centred scores read now (W_oh h_t). On a step where the new number gets
no lasting vote and is not a new largest, is p_{t+1}[0:9] = p_t[1:10] (votes move one slot toward
slot 0)? Measured on random inputs, slot 9 excluded (it holds the newest input)."""
import numpy as np
import torch as th

from algzoo.core import example, forward

g = th.Generator().manual_seed(17)
x = th.randn(200_000, 10, generator=g, dtype=th.float64)
_, H = forward(example(), x, return_hidden=True)
w_oh = example()[2]
p = H.numpy() @ w_oh.T
p -= p.mean(-1, keepdims=True)  # [B, 11, 10]
xn = x.numpy(); B, n = xn.shape
z = np.zeros((B, 2))
m2p = np.stack([np.sort(np.c_[z, xn[:, :t]], 1)[:, -2] for t in range(n)], 1)
m1p = np.stack([np.max(np.c_[z, xn[:, :t]], 1) for t in range(n)], 1)
kind = np.where(xn > m1p, "new largest", np.where((xn > 0.8 * m2p) & (xn > 0), "lasting", "quiet"))
print("step kind      n        error of the shift rule (rms / rms of belt)   belt's best slot moves by one")
for k in ("quiet", "lasting", "new largest"):
    errs, moves, cnt = [], [], 0
    for t in range(2, n):  # p index t+1 is after input t
        m = kind[:, t] == k
        before, after = p[m, t, 1:], p[m, t + 1, :9]
        e = np.sqrt(((after - before) ** 2).mean(1)) / np.sqrt((before ** 2).mean(1))
        errs.append(e)
        moves.append(after.argmax(1) == before.argmax(1))
        cnt += m.sum()
    e, mv = np.concatenate(errs), np.concatenate(moves)
    print(f"  {k:12s} {cnt:7d}       median {np.median(e):.3f}, 90th pct {np.percentile(e, 90):.3f}                  {mv.mean():.3f}")

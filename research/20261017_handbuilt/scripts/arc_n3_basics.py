"""ARC's neuron 3: weights, activity, and the effect of removing it."""
import numpy as np
import torch as th

from algzoo.core import example, forward

P = example()
w_hi, w_hh, w_oh = P
np.set_printoptions(precision=2, suppress=True, linewidth=200)
print("input weight", round(w_hi[3], 2), " self", round(w_hh[3, 3], 2))
print("weights into 3 :", w_hh[3])
print("weights out of 3 (to others):", w_hh[:, 3])
wc = w_oh - w_oh.mean(0)
print("output weights of 3 (centred):", wc[:, 3], " norm", np.linalg.norm(wc[:, 3]).round(1), " median norm of others", np.median(np.linalg.norm(wc, axis=0)).round(1))
g = th.Generator().manual_seed(0)
x = th.randn(1_000_000, 10, generator=g, dtype=th.float64)
_, H = forward(P, x, return_hidden=True)
H = H[:, 1:].numpy()
act = H[..., 3]
print("share of steps active, by step:", (act > 0).mean(0))
print("mean value when active:", act[act > 0].mean().round(2), " other neurons' mean when active:", np.round([H[..., k][H[..., k] > 0].mean() for k in range(16)], 1))
tgt = x.argsort(-1)[:, -2]
base = (forward(P, x).argmax(-1) == tgt).double().mean().item()
P0 = (w_hi.copy(), w_hh.copy(), w_oh.copy())
P0[2][:, 3] = 0
P1 = (w_hi.copy(), w_hh.copy(), w_oh.copy())
P1[1][:, 3] = 0; P1[2][:, 3] = 0  # neuron 3 silenced entirely (nobody reads it)
for lab, Q in (("output ignores 3", P0), ("3 silenced everywhere", P1)):
    print(f"accuracy normal {base:.4f}, {lab}: {(forward(Q, x).argmax(-1) == tgt).double().mean().item():.4f}")

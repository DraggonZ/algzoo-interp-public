"""Relabel and rescale the hand-built network so its heatmap lines up with the trained one. Only exact symmetries
are used, so the network computes the same function:
  - order: each hand-built neuron takes the index of the trained neuron with the same job; inside the clock group and
    the memory group, the trained neuron whose activity it tracks most closely (correlation over random lists;
    one-to-one assignment of labels);
  - loudness: each neuron is scaled (incoming weights x c, outgoing / c, c > 0) so it fires as loudly as its trained
    partner (RMS over the same lists). ReLU is unchanged by positive scaling.
No weight is fitted: c is a ratio of two measured activity sizes."""
import numpy as np
import torch as th
from scipy.optimize import linear_sum_assignment
from algzoo.core import example, forward
from algzoo.handbuilt_v18 import build_v18

T = [np.asarray(p, float) for p in example()]
import sys
net = build_v18(); I = net.idx
SRC = sys.argv[1] if len(sys.argv) > 1 else None
H = [np.asarray(p, float) for p in net.params] if SRC is None else [np.load(SRC)[k] for k in ("wi", "wh", "wo")]
OUT = sys.argv[2] if len(sys.argv) > 2 else "handbuilt_matched.npz"
x = th.randn(20000, 10, generator=th.Generator().manual_seed(0), dtype=th.float64)
def acts(P):
    lg, hs = forward(P, x, return_hidden=True)
    return lg.numpy(), hs[:, 1:].reshape(-1, len(P[0])).numpy()
lt, At = acts(T); lh, Ah = acts(H)

fixed = {"store": 2, "detect": 4, "keep": 5, "flag": 9}   # flip hides, so it joins the hiding group below
perm = {I[k]: v for k, v in fixed.items()}
def assign(ours, theirs):
    C = np.array([[abs(np.corrcoef(Ah[:, i], At[:, j])[0, 1]) if Ah[:, i].std() > 0 else 0 for j in theirs] for i in ours])
    r, c = linear_sum_assignment(-C)
    for a, b in zip(r, c):
        perm[ours[a]] = theirs[b]; print(f"  hand {ours[a]:2d} -> trained {theirs[b]:2d}  corr {C[a, b]:.2f}")
print("hiding:"); assign([I[f"clock{k}"] for k in range(3)] + [I["flip"]], [1, 6, 7, 8])
print("memory:"); assign([I[f"mem{j}"] for j in range(8)], [0, 3, 10, 11, 12, 13, 14, 15])
P = np.zeros(16, int)
for i, j in perm.items(): P[j] = i                      # new index j holds old neuron i
rms_t = np.sqrt((At ** 2).mean(0)); rms_h = np.sqrt((Ah ** 2).mean(0))
c = np.array([rms_t[j] / rms_h[P[j]] for j in range(16)])
wi, wh, wo = H[0][P] * c, H[1][np.ix_(P, P)] * c[:, None] / c[None, :], H[2][:, P] / c[None, :]
M = (wi, wh, wo)
lm, _ = acts(M)
print("same function: max logit difference", np.abs(lm - lh).max(), " accuracy", (lm.argmax(1) == x.numpy().argsort(1)[:, -2]).mean())
print("scales c:", np.array2string(c, precision=3))
np.savez(OUT, wi=wi, wh=wh, wo=wo, perm=P, scale=c)

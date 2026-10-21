"""Relabel and rescale twin16 (v23 with ARC's constants: U_mem 1, share 0.25, rho 0.99, level_exact False) into ARC's
neuron order and loudness, by exact symmetries only (same function), as paper/tweet/match_order_scale.py did for v18.
Order: store -> 2, detect -> 4, keep -> 5, the linear-route neuron -> 9 (ARC's late gate slot; the twin has no late
gate); the four hiding neurons (clock0-2, flip) -> ARC's 1, 6, 7, 8 and the eight memory slots -> ARC's memory, each by
one-to-one assignment on activity correlation. Loudness: incoming x c, outgoing / c, c = RMS ratio to the ARC partner."""
import numpy as np, torch as th
from scipy.optimize import linear_sum_assignment
from algzoo.core import example, forward
from algzoo.handbuilt_v23 import build_v23
T = [np.asarray(p, float) for p in example()]
net = build_v23(U_mem=1.0, share=0.25, rho=0.99, level_exact=False); H = [np.asarray(p, float) for p in net.params]; I = net.idx
x = th.randn(20000, 10, generator=th.Generator().manual_seed(0), dtype=th.float64)
def acts(P):
    lg, hs = forward(P, x, return_hidden=True); return lg.numpy(), hs[:, 1:].reshape(-1, len(P[0])).numpy()
lt, At = acts(T); lh, Ah = acts(H)
perm = {I["store"]: 2, I["detect"]: 4, I["keep"]: 5, I["linear"]: 9}
def assign(ours, theirs, lab):
    C = np.array([[abs(np.corrcoef(Ah[:, i], At[:, j])[0, 1]) if Ah[:, i].std() > 0 else 0 for j in theirs] for i in ours])
    r, c = linear_sum_assignment(-C)
    for a, b in zip(r, c): perm[ours[a]] = theirs[b]; print(f"  {lab}: hand {ours[a]:2d} -> ARC {theirs[b]:2d}  corr {C[a, b]:.2f}")
assign([I[f"clock{k}"] for k in range(3)] + [I["flip"]], [1, 6, 7, 8], "hiding")
assign([I[f"mem{j}"] for j in range(8)], [0, 3, 10, 11, 12, 13, 14, 15], "memory")
P = np.zeros(16, int)
for i, j in perm.items(): P[j] = i
c = np.sqrt((At ** 2).mean(0)) / np.sqrt((Ah ** 2).mean(0))[P]
M = (H[0][P] * c, H[1][np.ix_(P, P)] * c[:, None] / c[None, :], H[2][:, P] / c[None, :])
lm, _ = acts(M)
print("same function: max logit difference", np.abs(lm - lh).max())
X = th.randn(4_000_000, 10, generator=th.Generator().manual_seed(5), dtype=th.float64)
print("accuracy on 4M lists", ((forward(M, X).argmax(-1)) == X.argsort(-1)[:, -2]).double().mean().item())
np.savez("twin16_matched.npz", wi=M[0], wh=M[1], wo=M[2], perm=P, scale=c)

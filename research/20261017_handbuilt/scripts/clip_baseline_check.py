"""Do ARC's memory neurons run linearly on a baseline set by the largest so far?"""
import numpy as np, torch as th
from algzoo import core
np.set_printoptions(precision=2, suppress=True, linewidth=160)
p = core.example(); wi, wh, wo = p
X = th.randn(200_000, 10, generator=th.Generator().manual_seed(1), dtype=th.float64)
logits, H = core.forward(p, X, return_hidden=True)      # H: [N, 11, 16], H[:,t] = state after t inputs
H = H.numpy(); X = X.numpy()
m1 = np.maximum.accumulate(np.maximum(X, 0), axis=1)     # largest so far (>=0) after each input, [N,10]
mem = [0, 3, 10, 11, 12, 13, 14, 15]; canc = [1, 6, 7, 8]
sec = X.argsort(1)[:, -2]; ok = logits.numpy().argmax(1) == sec
print("accuracy", ok.mean())
# 1. how often is each memory neuron clipped (exactly zero) at steps 3..10, among lists whose largest so far > 0
mask = m1[:, 2:] > 0                                    # states after inputs 3..10 -> H[:, 3:]
act = H[:, 3:, :]
print("share of steps OFF (zero) per neuron, steps 3-10, largest so far > 0")
for grp, name in ((mem, "memory"), (canc, "cancellers"), ([2, 4, 5, 9], "store,det,keep,late")):
    print(f"  {name:22s}", np.array([(act[:, :, j][mask] == 0).mean() for j in grp]))
# 2. baseline: regress each memory neuron's activity (final step) on the final largest so far
A = H[:, 10, :]; M = m1[:, 9]
sel = M > 0.3
print("final activity per unit of the largest so far (slope, no intercept), memory neurons", mem)
sl = np.array([np.linalg.lstsq(M[sel, None], A[sel, j], rcond=None)[0][0] for j in mem]); print("  ", sl)
print("  R2 of that one-term fit per neuron", np.array([1 - ((A[sel, j] - sl[k] * M[sel]) ** 2).mean() / A[sel, j].var() for k, j in enumerate(mem)]))
# 3. is the baseline pattern's readout cancelled by the store's readout?
base_read = wo[:, mem] @ sl                             # score push per unit largest from the memory baseline
store_read = wo[:, 2] * np.linalg.lstsq(M[sel, None], A[sel, 2], rcond=None)[0][0]
print("push per unit largest, positions 0-9:")
print("  memory baseline ", base_read)
print("  store            ", store_read)
print("  sum              ", base_read + store_read, " (spread of sum / spread of baseline:", round((base_read + store_read).std() / base_read.std(), 2), ")")
# 4. the cancellers' common mode readout: push of a uniform unit on 1,6,7,8
print("  cancellers, uniform unit on all four", wo[:, canc].sum(1))
# 5. does clipping of memory neurons predict mistakes?
nclip = (act[:, :, mem] == 0).sum((1, 2))
for k in (0, 1, 2, 3):
    s = nclip == k if k < 3 else nclip >= 3
    print(f"  lists with {k if k<3 else '3+'} clipped memory-steps: share {s.mean():.3f}, accuracy {ok[s].mean():.3f}")

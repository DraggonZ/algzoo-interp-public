"""Additive votes fitted to ARC's choices (not its scores). Number at position t with type y contributes
x_t * W[y, t, k] to score k. Types: 0 final largest, 1 beaten record, 2 kept non-record, 3 forgotten, 4 last input
(if not the largest); types 1 and 2 split by the number of later bigger numbers (0, 1, 2, 3+).
Prints each type's own-position weight and its spill, to see what the network's effective votes look like."""
import numpy as np, torch as th
from algzoo.core import example, forward

dev = "cuda"
g = th.Generator().manual_seed(17)
x = th.randn(1_000_000, 10, generator=g, dtype=th.float64)
S = forward(example(), x).numpy()
xn = x.numpy(); B, n = xn.shape
z = np.zeros((B, 2))
m2p = np.stack([np.sort(np.c_[z, xn[:, :t]], 1)[:, -2] for t in range(n)], 1)
m1p = np.stack([np.max(np.c_[z, xn[:, :t]], 1) for t in range(n)], 1)
amax = xn.argmax(1)
lb = np.stack([(xn[:, t + 1:] > xn[:, t:t + 1]).sum(1) for t in range(n)], 1).clip(max=3)
record = xn > m1p
kept = record | ((xn >= 0.75 * m2p) & (xn > 0))
typ = np.where(record, 1 + lb, np.where(kept, 5 + lb, 9))          # 1-4 beaten record by lb, 5-8 kept, 9 forgotten
typ[np.arange(B), amax] = 0
typ[:, -1] = np.where(amax == n - 1, 0, 10)
T = 11
names = ["final largest"] + [f"old largest, {i}{'+' if i == 3 else ''} later bigger" for i in range(4)] + \
        [f"kept, {i}{'+' if i == 3 else ''} later bigger" for i in range(4)] + ["forgotten", "last input"]
F = np.zeros((B, T * n))
for t in range(n):
    F[np.arange(B), typ[:, t] * n + t] += xn[:, t]
tgt = xn.argsort(1)[:, -2]; arc = S.argmax(1)
tr, te = slice(0, 800_000), slice(800_000, None)
Ft = th.tensor(F, device=dev, dtype=th.float32); y = th.tensor(arc, device=dev)
W = th.zeros(T * n, n, device=dev, requires_grad=True)
opt = th.optim.Adam([W], lr=0.05)
for it in range(4000):
    idx = th.randint(0, 800_000, (65536,), device=dev)
    loss = th.nn.functional.cross_entropy(Ft[idx] @ W, y[idx])
    opt.zero_grad(); loss.backward(); opt.step()
with th.no_grad():
    P = (Ft[800_000:] @ W).argmax(1).cpu().numpy()
    W = W.cpu().numpy().reshape(T, n, n)
wrong = arc[te] != tgt[te]
print(f"fitted to ARC's choices: agrees {(P == arc[te]).mean():.3f}; accuracy {(P == tgt[te]).mean():.4f} (ARC {(arc[te] == tgt[te]).mean():.4f}); "
      f"ARC mistakes reproduced {(P[wrong] == arc[te][wrong]).mean():.2f}")
np.save("../votes/vote_choice_fit_W.npy", W)
counts = np.bincount(typ.ravel(), minlength=T) / typ.size
print("\nper type, averaged over positions 1-8: own-position weight; spill = mean |weight| elsewhere, as a share of own")
for y_ in range(T):
    own = np.array([W[y_, t, t] - W[y_, t].mean() for t in range(n)])
    spill = np.array([np.abs(np.delete(W[y_, t] - W[y_, t].mean(), t)).mean() for t in range(n)])
    print(f"  {names[y_]:32s} share {counts[y_]:.3f}  own by position {np.round(own, 1)}  spill/own (pos 1-8) {np.median(spill[1:9] / np.abs(own[1:9])):.2f}")

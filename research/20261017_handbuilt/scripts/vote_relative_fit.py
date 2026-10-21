"""Relative-vote model fitted to ARC's choices. Each number votes only on its own position:
  old largest (beaten record) at p:  a_rec[p] * x
  kept non-record at p:              a_kept[p] * x + b_kept[p] * second-so-far      (kept: x >= 0.6 * second-so-far, x > 0)
  forgotten at p:                    a_forg[p] * x
  last input (if not the largest):   a_last * x + b_last * second-so-far
Fitted by cross-entropy to ARC's answers. Prints per position: size (a) and discount (-b / a): is there fading,
and does the discount track it?"""
import numpy as np, torch as th
from algzoo.core import example, forward

dev = "cuda"
g = th.Generator().manual_seed(21)
x = th.randn(1_000_000, 10, generator=g, dtype=th.float64)
arc = forward(example(), x).numpy().argmax(1)
xn = x.numpy(); B, n = xn.shape
z = np.zeros((B, 2))
m2p = np.stack([np.sort(np.c_[z, xn[:, :t]], 1)[:, -2] for t in range(n)], 1)
m1p = np.stack([np.max(np.c_[z, xn[:, :t]], 1) for t in range(n)], 1)
record = xn > m1p; final = xn == xn.max(1, keepdims=True)
kept = ~record & (xn >= 0.6 * m2p) & (xn > 0)
tgt = xn.argsort(1)[:, -2]
P = np.arange(n)
feats = []  # each (B, n) array placed on the own position
names = []
for name, mask, val in (("rec_x", record & ~final, xn), ("kept_x", kept, xn), ("kept_s", kept, m2p),
                        ("forg_x", ~record & ~kept, xn)):
    feats.append(np.where(mask, val, 0.0)); names.append(name)
F = th.tensor(np.stack(feats, 0), device=dev, dtype=th.float32)          # (4, B, n)
y = th.tensor(arc, device=dev)
W = th.zeros(4, n, device=dev, requires_grad=True)
with th.no_grad(): W[0] += 1; W[1] += 1
opt = th.optim.Adam([W], lr=0.05)
tr = 800_000
for it in range(3000):
    idx = th.randint(0, tr, (65536,), device=dev)
    s = (F[:, idx] * W[:, None, :]).sum(0)
    loss = th.nn.functional.cross_entropy(s * 1.0, y[idx])
    opt.zero_grad(); loss.backward(); opt.step()
with th.no_grad():
    s = (F[:, tr:] * W[:, None, :]).sum(0)
    pred = s.argmax(1).cpu().numpy(); Wn = W.cpu().numpy()
te = slice(tr, None); w = arc[te] != tgt[te]
print(f"fit: accuracy {(pred == tgt[te]).mean():.4f} (ARC {(arc[te] == tgt[te]).mean():.4f}); agrees {(pred == arc[te]).mean():.3f}; "
      f"ARC mistakes reproduced {(pred[w] == arc[te][w]).mean():.2f}")
np.set_printoptions(precision=2, suppress=True, linewidth=150)
sc = Wn[1, 5]
print("sizes relative to a kept number at position 5")
for i, nm in enumerate(names):
    print(f"  {nm:7s}", Wn[i] / sc)
print("  discount -b/a for kept numbers by position:", -Wn[2] / Wn[1])
np.save("../votes/vote_relative_fit_W.npy", Wn)

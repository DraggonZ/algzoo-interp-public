"""Decision-level spill of the revealed vote. Vote model fitted to ARC's answers (cross-entropy), as in
vote_relative_fit.py, plus spill: an old largest at p also votes on positions p-2..p+2 with fitted weights (shared
over p). Same for kept numbers. Reports fitted spill relative to the own-position weight, and agreement."""
import numpy as np, torch as th
from algzoo.core import example, forward

dev = "cuda"
g = th.Generator().manual_seed(22)
x = th.randn(1_000_000, 10, generator=g, dtype=th.float64)
arc = forward(example(), x).numpy().argmax(1)
xn = x.numpy(); B, n = xn.shape
z = np.zeros((B, 2))
m2p = np.stack([np.sort(np.c_[z, xn[:, :t]], 1)[:, -2] for t in range(n)], 1)
m1p = np.stack([np.max(np.c_[z, xn[:, :t]], 1) for t in range(n)], 1)
record = xn > m1p; final = xn == xn.max(1, keepdims=True)
kept = ~record & (xn >= 0.6 * m2p) & (xn > 0)
tgt = xn.argsort(1)[:, -2]
offs = [-2, -1, 1, 2]
# features: own-position (per position) for rec, kept_x, kept_s, forg; spill (per offset, shared) for rec and kept
own = [np.where(record & ~final, xn, 0), np.where(kept, xn, 0), np.where(kept, m2p, 0), np.where(~record & ~kept, xn, 0)]
F_own = th.tensor(np.stack(own, 0), device=dev, dtype=th.float32)          # (4, B, n)
def spill(mask):
    out = np.zeros((len(offs), B, n))
    for i, o in enumerate(offs):
        for p in range(n):
            q = p + o
            if 0 <= q < 9: out[i, :, q] += np.where(mask[:, p], xn[:, p], 0)
    return out
F_sp = th.tensor(np.concatenate([spill(record & ~final), spill(kept)], 0), device=dev, dtype=th.float32)  # (8, B, n)
y = th.tensor(arc, device=dev)
W = th.zeros(4, n, device=dev); W[0] += 1; W[1] += 1; W.requires_grad_(True)
S = th.zeros(8, device=dev, requires_grad=True)
opt = th.optim.Adam([W, S], lr=0.05)
tr = 800_000
for it in range(3000):
    idx = th.randint(0, tr, (65536,), device=dev)
    sc = (F_own[:, idx] * W[:, None, :]).sum(0) + (F_sp[:, idx] * S[:, None, None]).sum(0)
    loss = th.nn.functional.cross_entropy(sc, y[idx])
    opt.zero_grad(); loss.backward(); opt.step()
with th.no_grad():
    sc = (F_own[:, tr:] * W[:, None, :]).sum(0) + (F_sp[:, tr:] * S[:, None, None]).sum(0)
    pred = sc.argmax(1).cpu().numpy()
te = slice(tr, None)
print(f"agree with ARC {(pred == arc[te]).mean():.3f}; accuracy {(pred == tgt[te]).mean():.4f}")
Wn, Sn = W.detach().cpu().numpy(), S.detach().cpu().numpy()
rec_own, kept_own = Wn[0, 1:8].mean(), Wn[1, 1:8].mean()
print("old largest: spill at offsets -2,-1,+1,+2 as share of its own weight:", np.round(Sn[:4] / rec_own, 2))
print("kept number: spill at offsets -2,-1,+1,+2 as share of its own weight:", np.round(Sn[4:] / kept_own, 2))

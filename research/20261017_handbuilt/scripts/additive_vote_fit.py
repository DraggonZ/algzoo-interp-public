"""Best additive vote model of ARC's scores. Each number i contributes x_i * W[slot j, type_i, age_i]
to slot j's score. Types (from the account): 0 current largest, 1 lasting (beat 0.8 of the second
so far, or was a record) , 2 no lasting vote, 3 last input. Variants add: number of later bigger
numbers (shrinkage) as extra type split. Fit by least squares on centred scores (train), score
argmax agreement with ARC and accuracy on held-out inputs."""
import numpy as np
import torch as th

from algzoo.core import example, forward

g = th.Generator().manual_seed(16)
x = th.randn(400_000, 10, generator=g, dtype=th.float64)
S = forward(example(), x).numpy(); S -= S.mean(1, keepdims=True)
xn = x.numpy(); B, n = xn.shape
z = np.zeros((B, 2))
m2p = np.stack([np.sort(np.c_[z, xn[:, :t]], 1)[:, -2] for t in range(n)], 1)
m1p = np.stack([np.max(np.c_[z, xn[:, :t]], 1) for t in range(n)], 1)
amax = xn.argmax(1)
later_bigger = np.stack([(xn[:, t + 1:] > xn[:, t:t + 1]).sum(1) for t in range(n)], 1)


def features(variant):
    typ = np.where((xn > 0.8 * m2p) & (xn > 0) | (xn > m1p), 1, 2)
    typ[np.arange(B), amax] = 0
    typ[:, -1] = np.where(amax == n - 1, 0, 3)
    if variant == "shrink":
        typ = np.where(typ == 1, 1 + 4 * np.minimum(later_bigger, 3), typ)  # 1,5,9,13
    if variant == "pos":  # also split values into positive / negative parts
        pass
    T = typ.max() + 1
    F = np.zeros((B, T * n))
    for t in range(n):
        F[np.arange(B), typ[:, t] * n + (n - 1 - t)] += xn[:, t]  # type x age
    if variant == "pos":
        F2 = np.zeros((B, T * n))
        for t in range(n):
            F2[np.arange(B), typ[:, t] * n + (n - 1 - t)] += np.maximum(xn[:, t], 0)
        F = np.c_[F, F2]
    return F


tr, te = slice(0, 300_000), slice(300_000, None)
tgt = xn.argsort(1)[:, -2]
arc = S.argmax(1)
for variant in ("base", "shrink", "pos"):
    F = features(variant)
    W, *_ = np.linalg.lstsq(F[tr], S[tr], rcond=None)
    P = F[te] @ W
    r2 = 1 - ((P - S[te]) ** 2).sum() / (S[te] ** 2).sum()
    agree = (P.argmax(1) == arc[te]).mean()
    acc = (P.argmax(1) == tgt[te]).mean()
    wrong = arc[te] != tgt[te]
    print(f"{variant:7s}: {F.shape[1]:3d} features per slot; score R^2 {r2:.3f}; agrees with ARC {agree:.3f}; "
          f"accuracy {acc:.3f} (ARC {(arc[te]==tgt[te]).mean():.3f}); ARC mistakes reproduced {(P.argmax(1)[wrong]==arc[te][wrong]).mean():.2f}")

# fit to ARC's choices: softmax over slots with additive vote scores (cross-entropy on ARC's argmax)
dev = "cuda"
for variant in ("base", "shrink"):
    F = th.tensor(features(variant), device=dev, dtype=th.float32)
    y = th.tensor(arc, device=dev)
    W = th.zeros(F.shape[1], n, device=dev, requires_grad=True)
    opt = th.optim.Adam([W], lr=0.05)
    for it in range(3000):
        idx = th.randint(0, 300_000, (65536,), device=dev)
        loss = th.nn.functional.cross_entropy(F[idx] @ W * 1.0, y[idx])
        opt.zero_grad(); loss.backward(); opt.step()
    with th.no_grad():
        P = (F[300_000:] @ W).argmax(1).cpu().numpy()
    wrong = arc[te] != tgt[te]
    print(f"{variant:7s} fitted to ARC's choices: agrees with ARC {(P == arc[te]).mean():.3f}; accuracy {(P == tgt[te]).mean():.3f}; "
          f"ARC mistakes reproduced {(P[wrong] == arc[te][wrong]).mean():.2f}")

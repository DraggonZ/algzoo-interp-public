"""Shared pieces for the Level 2 noise study: networks, noisy copies, true accuracy by sampling (batched over copies)."""
import sys
import numpy as np
import torch as th
sys.path.insert(0, "/data/output/research/algzoo/20261017_handbuilt/family")
from algzoo.core import example, load

DEV = "cuda"; DT = th.float64
th.set_num_threads(2)


def net(name):
    if name == "ARC":
        W = example()
    else:
        W = load(f"/data/output/research/algzoo/20261016_run3/models/2nd_argmax_16_10_1073741824_{name}.pth")
    W = W.params if hasattr(W, "params") else W
    return [th.as_tensor(np.asarray(w, float), dtype=DT, device=DEV) for w in W]


def flat(W):
    return th.cat([w.reshape(-1) for w in W])


def unflat(v, W):
    """v: [C, 432] -> list of [C, *shape]."""
    out, i = [], 0
    for w in W:
        out.append(v[:, i:i + w.numel()].reshape(len(v), *w.shape)); i += w.numel()
    return out


def arc_hilton_copies():
    """The 200 noise draws of hilton_noise.py, regenerated exactly (generator 41: 100,000 lists first, then the draws)."""
    g = th.Generator(device=DEV).manual_seed(41)
    X = th.randn(100000, 10, device=DEV, dtype=DT, generator=g)
    dws = th.stack([th.randn(432, device=DEV, dtype=DT, generator=g) * 0.0025 for _ in range(200)])
    return X, dws


def draws(n, sig, seed):
    g = th.Generator(device=DEV).manual_seed(seed)
    return th.randn(n, 432, device=DEV, dtype=DT, generator=g) * sig


def accuracy(W, dws, n_lists=1_000_000, seed=12345, chunk=250_000, cb=25):
    """True accuracy of each copy W + dw on the same n_lists random lists. Returns numpy [C]."""
    C = len(dws); correct = th.zeros(C, device=DEV, dtype=DT)
    g = th.Generator(device=DEV).manual_seed(seed)
    for b in range(0, n_lists, chunk):
        X = th.randn(min(chunk, n_lists - b), 10, device=DEV, dtype=DT, generator=g); y = X.argsort(-1)[:, -2]
        for c0 in range(0, C, cb):
            wi, wh, wo = unflat(flat(W)[None] + dws[c0:c0 + cb], W)       # [c,16], [c,16,16], [c,10,16]
            h = th.zeros(len(wi), len(X), 16, device=DEV, dtype=DT)
            for t in range(10):
                h = th.relu(th.einsum("cbj,cij->cbi", h, wh) + X[None, :, t:t + 1] * wi[:, None, :])
            s = th.einsum("cbj,cij->cbi", h, wo)
            correct[c0:c0 + cb] += (s.argmax(-1) == y[None]).sum(1).to(DT)
    return (correct / n_lists).cpu().numpy()

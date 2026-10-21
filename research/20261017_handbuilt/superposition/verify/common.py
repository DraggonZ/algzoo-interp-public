"""Shared helpers for the independent verification (float64, torch on GPU)."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np
import torch

torch.set_num_threads(2)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
DT = torch.float64
MODELS = "/data/output/research/algzoo/20261016_run3/models/2nd_argmax_16_10_1073741824_{}.pth"


def to_t(params):
    return tuple(torch.as_tensor(np.asarray(p), dtype=DT, device=DEV) for p in params)


def arc():
    from algzoo.core import example
    return to_t(example())


def v18():
    from algzoo.handbuilt_v18 import build_v18
    return to_t(build_v18().params)


def others():
    from algzoo.core import load
    return {s: to_t(load(MODELS.format(s))) for s in list(range(100, 116)) + list(range(200, 216))}


def scores(params, x, step_noise=None, final_noise=None):
    """x: [B,10]. h_t = relu(W_hh h + w_hi x_t); scores = W_oh h_10.
    step_noise: optional [10,B,16] added to the state after each relu.
    final_noise: optional [B,16] added to h_10 only."""
    w_hi, W_hh, W_oh = params
    h = torch.zeros(x.shape[0], W_hh.shape[0], dtype=x.dtype, device=x.device)
    for t in range(x.shape[1]):
        h = torch.relu(h @ W_hh.T + x[:, t:t + 1] * w_hi)
        if step_noise is not None:
            h = h + step_noise[t]
    if final_noise is not None:
        h = h + final_noise
    return h @ W_oh.T


def states(params, x):
    w_hi, W_hh, W_oh = params
    h = torch.zeros(x.shape[0], W_hh.shape[0], dtype=x.dtype, device=x.device)
    out = []
    for t in range(x.shape[1]):
        h = torch.relu(h @ W_hh.T + x[:, t:t + 1] * w_hi)
        out.append(h)
    return torch.stack(out)  # [10,B,16]


def predict(params, x, chunk=500_000, **kw):
    return torch.cat([scores(params, x[i:i + chunk], **kw).argmax(1) for i in range(0, len(x), chunk)])


def target(x):
    return x.argsort(1, descending=True)[:, 1]


def flat(params):
    return torch.cat([p.reshape(-1) for p in params])


def unflat(v, like):
    out, i = [], 0
    for p in like:
        out.append(v[i:i + p.numel()].reshape(p.shape)); i += p.numel()
    return tuple(out)


def draw(n, gen):
    return torch.randn(n, 10, dtype=DT, device=DEV, generator=gen)


def gen(seed):
    g = torch.Generator(device=DEV); g.manual_seed(seed); return g


def se(p, n):
    return 100 * (p * (1 - p) / n) ** 0.5

"""Minimal loader and evaluator for AlgZoo 2nd-argmax RNNs.

Model: h_0 = 0, h_{t+1} = ReLU(W_hh h_t + W_hi x_t), logits = W_oh h_n.
No biases, so the model is positively homogeneous in x.
"""
from pathlib import Path

import torch as th

from algzoo.paths import MODELS


def load(name):
    """Return (W_hi[d], W_hh[d,d], W_oh[n,d]) as float64 numpy arrays."""
    path = Path(name)
    if not path.exists():
        for sub in ("extra", "zoo"):
            if (MODELS / sub / name).exists():
                path = MODELS / sub / name
                break
    sd = th.load(path, weights_only=True, map_location="cpu")
    w_hi = sd["rnn.weight_ih_l0"].double().numpy()[:, 0]
    w_hh = sd["rnn.weight_hh_l0"].double().numpy()
    w_oh = sd["linear.weight"].double().numpy()
    return w_hi, w_hh, w_oh


def example():
    return load("one_layer_16_hidden_94_acc_2nd_argmax.pth")


def zoo(d, n, seed=None, n_seqs=2**30):
    if seed is None:
        return load(f"2nd_argmax_{d}_{n}_{n_seqs}_bo5.pth")
    return load(f"2nd_argmax_{d}_{n}_{n_seqs}_{seed}.pth")


def forward(params, x, return_hidden=False):
    """x: torch tensor [..., n] on any device; params: numpy triple."""
    w_hi, w_hh, w_oh = (th.as_tensor(p, dtype=x.dtype, device=x.device) for p in params)
    h = th.zeros(*x.shape[:-1], w_hh.shape[0], dtype=x.dtype, device=x.device)
    hs = [h]
    for t in range(x.shape[-1]):
        h = th.relu(h @ w_hh.T + x[..., t, None] * w_hi)
        hs.append(h)
    logits = h @ w_oh.T
    if return_hidden:
        return logits, th.stack(hs, -2)
    return logits


def target(x):
    return x.argsort(-1)[..., -2]


@th.no_grad()
def accuracy(params, n_samples=10**8, batch=2**22, seed=0, device="cuda", dtype=th.float32):
    """Monte Carlo accuracy on standard Gaussian inputs. Returns (acc, n)."""
    n = params[2].shape[0]
    g = th.Generator(device=device).manual_seed(seed)
    correct = 0
    done = 0
    while done < n_samples:
        b = min(batch, n_samples - done)
        x = th.randn(b, n, generator=g, device=device, dtype=dtype)
        correct += (forward(params, x).argmax(-1) == target(x)).sum().item()
        done += b
    return correct / done, done

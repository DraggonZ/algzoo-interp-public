"""The 38 networks of the universality study and shared helpers.
ARC's studied network, ARC's zoo seeds 0-4 and our run-3 seeds 100-115, 200-215 (all 16 neurons, 10 steps, no biases).
Weights are (w_in [16], W_hh [16, 16] with W[i, j] = weight from j to i, W_out [10, 16])."""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, "/data/output/src/algzoo/src")
import numpy as np, torch
from algzoo.core import load, example
DEV = "cuda"; DT = torch.float64
ROOT = "/data/output/research/algzoo"
ZOO = ROOT + "/models/zoo/2nd_argmax_16_10_%d_%d.pth"
RUN3 = ROOT + "/20261016_run3/models/2nd_argmax_16_10_1073741824_%d.pth"
CHECKPOINTS = [0] + [2 ** k for k in range(20, 31)]          # training lists seen; 0 = initial weights

def np_w(p): return [np.asarray(a, float) for a in p]
def networks():
    """Ordered dict label -> numpy weights. ARC first."""
    out = {"ARC": np_w(example())}
    for s in range(5): out[f"zoo{s}"] = np_w(load(ZOO % (2 ** 30, s)))
    for s in list(range(100, 116)) + list(range(200, 216)): out[f"s{s}"] = np_w(load(RUN3 % s))
    return out
def zoo_checkpoint(seed, n): return np_w(load(ZOO % (n, seed)))
def T(W): return [torch.tensor(a, device=DEV, dtype=DT) for a in W]
def lists(n, seed):
    return torch.randn(n, 10, device=DEV, dtype=DT, generator=torch.Generator(device=DEV).manual_seed(seed))
def run(W, X):
    """States after each step, [n, 10, H]; H[:, t] has read x_0..x_t. W: torch weights."""
    h = torch.zeros(len(X), len(W[0]), device=DEV, dtype=DT); H = []
    for t in range(10):
        h = torch.relu(h @ W[1].T + X[:, t:t + 1] * W[0]); H.append(h)
    return torch.stack(H, 1)
def records(X):
    """[n, 10] bool: x_t positive and above every earlier number, t >= 1."""
    cm = X.cummax(1).values
    prev = torch.cat([torch.full((len(X), 1), -1e9, device=DEV, dtype=DT), cm[:, :-1]], 1)
    r = (X > prev) & (X > 0); r[:, 0] = False
    return r
def gauge(W, H):
    """Rescale each neuron so its activity has standard deviation 1 over all steps (one fixed choice of the free scales).
    Returns numpy weights in that gauge and the scales sigma."""
    sig = H.reshape(-1, H.shape[-1]).std(0).cpu().numpy(); sig = np.where(sig > 1e-12, sig, 1.0)
    wi, wh, wo = W
    return [wi / sig, wh * sig[None, :] / sig[:, None], wo * sig[None, :]], sig

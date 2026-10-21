"""Shared pieces for the estimate study (2026-10-21). Torch on the GPU, float64.
Networks: h_{t+1} = ReLU(W_hh h_t + w_in x_t), scores = W_out h_10, no biases.
cells(X): per list and step, an order cell: step, ages of the three largest positive numbers so far (-1 if fewer),
sign of the arriving number. Cell index fits in 10*11*11*11*2.
run(W, X, on_fn): the network, or a surrogate whose on/off pattern comes from on_fn(t, pre) -> bool [n,16]."""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, "/data/output/src/algzoo/src")
import numpy as np, torch
from algzoo.core import example, load

DEV = "cuda"; DT = torch.float64
NCELL = 10 * 11 * 11 * 11 * 2
TRUE_ARC = 95.32
RUN3 = "/data/output/research/algzoo/20261016_run3/models/2nd_argmax_16_10_1073741824_%d.pth"
GROUPS_ARC = {"memory": [0, 3, 10, 11, 12, 13, 14, 15], "hiding": [1, 6, 7, 8], "store": [2], "detector": [4],
              "keep gate": [5], "late gate": [9]}


def tw(p): return [torch.tensor(np.asarray(a, float), device=DEV, dtype=DT) for a in p]
def arc(): return tw(example())
def sister(s): return tw(load(RUN3 % s))


def lists(n, seed):
    return torch.randn(n, 10, device=DEV, dtype=DT, generator=torch.Generator(device=DEV).manual_seed(seed))


def ages(X):
    """[n,10,3] ages of the three largest positive numbers among x_0..x_t (-1 if fewer positive)."""
    n = len(X); out = torch.full((n, 10, 3), -1, device=DEV, dtype=torch.long)
    for t in range(10):
        xs = torch.where(X[:, :t + 1] > 0, X[:, :t + 1], torch.tensor(-np.inf, device=DEV, dtype=DT))
        k = min(3, t + 1); v, i = xs.topk(k, 1)
        out[:, t, :k] = torch.where(torch.isfinite(v), t - i, torch.full_like(i, -1))
    return out


def cells(X):
    a = ages(X) + 1; t = torch.arange(10, device=DEV)[None, :]
    return ((((t * 11 + a[..., 0]) * 11 + a[..., 1]) * 11 + a[..., 2]) * 2 + (X > 0).long())


def run(W, X, on_fn=None):
    """Returns scores [n,10] and the on/off pattern [n,10,H]."""
    wi, wh, wo = W; h = torch.zeros(len(X), len(wi), device=DEV, dtype=DT); ons = []
    for t in range(10):
        pre = h @ wh.T + X[:, t:t + 1] * wi
        on = pre > 0 if on_fn is None else on_fn(t, pre)
        h = pre * on; ons.append(on)
    return h @ wo.T, torch.stack(ons, 1)


def acc(S, X):
    return 100 * (S.argmax(1) == X.argsort(1)[:, -2]).double().mean().item()


def fit_table(W, X, C=None):
    """Majority on/off per cell and neuron from the network's own run. Returns [NCELL, H] bool and the cell counts."""
    _, on = run(W, X); C = cells(X) if C is None else C; H = on.shape[-1]
    cnt = torch.zeros(NCELL, device=DEV, dtype=DT).index_add_(0, C.reshape(-1), torch.ones(C.numel(), device=DEV, dtype=DT))
    s = torch.zeros(NCELL, H, device=DEV, dtype=DT).index_add_(0, C.reshape(-1), on.reshape(-1, H).double())
    return (s > cnt[:, None] / 2) | (cnt[:, None] == 0), cnt

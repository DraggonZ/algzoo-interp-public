"""P1-P3 of README: exactness against v19, single-record lists, and accuracy against the wrap weight."""
import numpy as np
import torch
from algzoo.handbuilt_v19 import build_v19
from algzoo.handbuilt_v23_from_v19 import build_v23f

DEV, dt = "cuda", torch.float64


def T(net):
    return [torch.tensor(np.asarray(p, float), device=DEV, dtype=dt) for p in net.params]


def scores(W, X):
    h = torch.zeros(len(X), len(W[0]), device=DEV, dtype=dt)
    for t in range(X.shape[1]):
        h = torch.relu(h @ W[1].T + X[:, t:t + 1] * W[0])
    return h @ W[2].T


def lists(n, seed):
    g = torch.Generator(device=DEV).manual_seed(seed)
    return torch.randn(n, 10, device=DEV, dtype=dt, generator=g)


def acc(W, X):
    s = X.argsort(1, descending=True)[:, 1]
    return (scores(W, X).argmax(1) == s).double().mean().item()


if __name__ == "__main__":
    X = lists(10_000, 1)
    v19 = scores(T(build_v19(kappa=0.5)), X)
    for ring in (False, True):
        new = scores(T(build_v23f(ep=0.0, r1=0.0, gate_level=0.5, ring=ring)), X)
        print(f"P1 ring={ring}: max |v23f - v19| = {(new - v19).abs().max().item():.2e} (v19 scores up to {v19.abs().max().item():.1f})")
    X = lists(1_000_000, 2)
    cm = torch.cummax(torch.cat([torch.zeros(len(X), 1, device=DEV, dtype=dt), X], 1), 1).values
    nrec = (X > cm[:, :-1]).sum(1)
    one = nrec == 1
    print(f"lists with a single record: {100*one.double().mean().item():.1f}%")
    base = scores(T(build_v23f(ep=0.05)), X[one]).argmax(1)
    for ep in (0.1, 0.3, 1.0):
        a = scores(T(build_v23f(ep=ep)), X[one]).argmax(1)
        print(f"P2 wrap {ep}: single-record answers equal to wrap 0.05 on {100*(a == base).double().mean().item():.3f}%")
    print(f"v19 (kappa 0.5): {100*acc(T(build_v19(kappa=0.5)), X):.2f}%")
    for ep in (0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0):
        print(f"P3 wrap {ep:4.2f} (ring fade {ep**(1/8):.3f} per step): {100*acc(T(build_v23f(ep=ep)), X):.2f}%", flush=True)

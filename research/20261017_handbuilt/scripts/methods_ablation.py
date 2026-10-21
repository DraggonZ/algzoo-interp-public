"""Methods study 1: do zero, mean and resample ablation recover the planted roles in the hand-built
network (gains 100)? Per neuron: accuracy overall and by the second's position.
Planted truth: 'last' matters only when the answer is position 9; clockA/clockB/vote carry the answer for
positions 0-8; store/detect/gates are needed everywhere. Same table for ARC's model for comparison."""
import numpy as np
import torch as th

from algzoo import handbuilt as hb
from algzoo.core import example

g = th.Generator(device="cuda").manual_seed(31)
X = th.randn(1_000_000, 10, generator=g, device="cuda", dtype=th.float64)
tgt = X.argsort(-1)[:, -2]


def run(P, mode=None, k=None, mean=None, donor=None):
    wi, wh, wo = (th.tensor(p, device="cuda") for p in P)
    h = th.zeros(len(X), len(wi), device="cuda", dtype=X.dtype)
    hd = th.zeros_like(h)
    for t in range(10):
        h = th.relu(h @ wh.T + X[:, t, None] * wi)
        if mode == "resample":
            hd = th.relu(hd @ wh.T + donor[:, t, None] * wi)
            h[:, k] = hd[:, k]
        elif mode == "zero":
            h[:, k] = 0
        elif mode == "mean":
            h[:, k] = mean[t, k]
        if mode is None:
            yield h.clone()
    if mode is not None:
        yield (h @ wo.T).argmax(-1)


def table(name, P, labels):
    hs = list(run(P)); mean = th.stack([h.mean(0) for h in hs])
    wo = th.tensor(P[2], device="cuda"); base = (hs[-1] @ wo.T).argmax(-1)
    donor = X[th.randperm(len(X), device="cuda")]
    print(f"\n{name}: accuracy {(base == tgt).double().mean():.3f}. Per neuron: accuracy after ablation (zero / mean / resample); for zero ablation also the share right when the answer is position 9 vs 0-8")
    for k, lab in enumerate(labels):
        accs = []
        for mode in ("zero", "mean", "resample"):
            pred = next(run(P, mode, k, mean, donor))
            accs.append((pred == tgt).double().mean().item())
            if mode == "zero":
                pz = pred
        a9 = (pz[tgt == 9] == 9).double().mean().item(); a08 = (pz[tgt != 9] == tgt[tgt != 9]).double().mean().item()
        print(f"  {lab:9s} {accs[0]:.3f} / {accs[1]:.3f} / {accs[2]:.3f}    zero: answer at 9 {a9:.2f}, at 0-8 {a08:.2f}")


net = hb.build_v2(100.0, 100.0)
labels = [None] * 16
for nm, i in net.idx.items():
    labels[i] = nm
table("Hand-built (gains 100)", net.params, labels)
table("ARC", example(), [f"n{k}" for k in range(16)])

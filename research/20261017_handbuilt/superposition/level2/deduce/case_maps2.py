"""Refine the one-map-per-arrangement estimate: one canonical list per arrangement and per cell of a grid over the two
ratios that the account's switching depends on (second/largest, third/second). Cell edges at equal-probability
quantiles of the input distribution, canonical values at cell medians. Accuracy error against cost."""
import sys, itertools; sys.path.insert(0, "../../../family")
import numpy as np, torch
from scipy.stats import norm
from common import *
from case_maps import A, idx
g = torch.Generator(device=DEV).manual_seed(11); Z = torch.randn(4_000_000, 10, device=DEV, dtype=dt, generator=g).sort(1, descending=True).values
qL = Z[:, 0].median().item()                       # input distribution only (order statistics), not the network
def cells(k):
    r1 = (Z[:, 1] / Z[:, 0]); r2 = (Z[:, 2] / Z[:, 1])
    e1 = torch.quantile(r1[:1_000_000], torch.linspace(0, 1, k + 1, device=DEV, dtype=dt)); e2 = torch.quantile(r2[:1_000_000], torch.linspace(0, 1, k + 1, device=DEV, dtype=dt))
    m1 = torch.quantile(r1[:1_000_000], (torch.arange(k, device=DEV, dtype=dt) + 0.5) / k); m2 = torch.quantile(r2[:1_000_000], (torch.arange(k, device=DEV, dtype=dt) + 0.5) / k)
    return e1, e2, m1, m2
def run(W, k, X):
    e1, e2, m1, m2 = cells(k); C = []
    for (l, s, t) in A:
        for i in range(k):
            for j in range(k):
                xs = qL * m1[i].item(); xt = xs * m2[j].item(); rest = -norm.pdf(xt) / norm.cdf(xt)
                c = [rest] * 10; c[l], c[s], c[t] = qL, xs, xt; C.append(c)
    C = torch.tensor(C, device=DEV, dtype=dt); J = torch.cat([jac(W, C[i:i + 50000]) for i in range(0, len(C), 50000)])
    v = X.sort(1, descending=True).values; l, s, t = top3(X)
    a = torch.tensor([idx[q] for q in zip(l.tolist(), s.tolist(), t.tolist())], device=DEV)
    i = torch.bucketize(v[:, 1] / v[:, 0], e1[1:-1].contiguous()); j = torch.bucketize(v[:, 2] / v[:, 1], e2[1:-1].contiguous())
    # negative ratios when values are negative: bucketize still assigns an edge cell
    cell = a * k * k + i * k + j
    pred = torch.einsum("noi,ni->no", J[cell], X).argmax(1)
    return (pred == s).double().mean().item(), len(C)
if __name__ == "__main__":
    nets = [("ARC", ARC)] + [(f"seed {s}", trained(s)) for s in (100, 104, 107, 111, 114, 209, 211)]
    gg = torch.Generator(device=DEV).manual_seed(3); X = torch.randn(500_000, 10, device=DEV, dtype=dt, generator=gg)
    s_true = top3(X)[1]
    for name, W in nets:
        tr = (scores(W, X).argmax(1) == s_true).double().mean().item(); row = []
        for k in (1, 2, 4, 8):
            acc, n = run(W, k, X); row.append(f"k {k}: {100*(acc-tr):+6.2f} ({n*11:,} runs)")
        print(f"{name:9s} true {100*tr:.2f}% | error of deduced estimate, points: " + " | ".join(row), flush=True)

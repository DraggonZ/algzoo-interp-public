"""r4a analysis 1: does a 2-D growing hiding wave (no flip neuron) keep the twin working?
Own accuracy on 1M lists (seed 1000) of: the twin's rules with v23's flip wave vs the 2-D wave, without and with r3b's
runner-up comparator; and the full r4a build (2-D wave + runner-up + late gate). Also the spill values g(k)."""
import sys
import numpy as np, torch
import algzoo.handbuilt_v23 as v23
from algzoo.handbuilt_v27b import build_v27b
from algzoo.handbuilt_v4a import build_v4a, wave_2d, g_of

DEV, DT = "cuda", torch.float64


def acc(net, n=1_000_000, seed=1000):
    wi, wh, wo = (torch.tensor(np.asarray(p, float), device=DEV, dtype=DT) for p in net.params)
    X = torch.randn(n, 10, device=DEV, dtype=DT, generator=torch.Generator(device=DEV).manual_seed(seed))
    h = torch.zeros(n, len(wi), device=DEV, dtype=DT)
    for t in range(10):
        h = torch.relu(h @ wh.T + X[:, t:t + 1] * wi)
    return 100 * ((h @ wo.T).argmax(1) == X.argsort(1, descending=True)[:, 1]).double().mean().item()


A, b, c = wave_2d()
print("g(k), k = -9..8:", " ".join(f"{k}:{g_of(A, b, c, k):.3g}" for k in range(-9, 9)))
base = v23.build_v23(w=0.0, U_mem=1.0, share=0.25, rho=0.99, level_exact=True)
base.params[1][base.idx["linear"], base.idx["detect"]] += 0.1 / 1000
print(f"twin rules, flip wave, position-0 cell (eps 0.1):        {acc(base):.2f}")
print(f"twin rules, 2-D wave, position-0 cell (eps 0.1 via L):   {acc(build_v4a(q=False, late=False)):.2f}")
print(f"r3b (flip wave + runner-up + keep cut), exact level:      {acc(build_v27b(level_exact=True)):.2f}")
print(f"2-D wave + runner-up + keep cut, push via L, no late gate: {acc(build_v4a(q=True, late=False)):.2f}")
print(f"r4a full: 2-D wave + runner-up + keep cut + late gate:     {acc(build_v4a(q=True, late=True)):.2f}")
for rw in (3.0, 5.0):
    print(f"r4a full, rho_w {rw}:                                    {acc(build_v4a(rho_w=rw)):.2f}")

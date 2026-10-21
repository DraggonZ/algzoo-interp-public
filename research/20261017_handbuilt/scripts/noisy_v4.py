"""Exact estimate (bias_formula_v4) on noisy copies of ARC's network, term by term against measured mistake kinds
(same noise draws and lists as noisy_kinds.py)."""
import sys
from pathlib import Path

import numpy as np
import torch as th

sys.path.insert(0, str(Path(__file__).parent))
from bias_formula_v4 import estimate  # noqa: E402

from algzoo.core import example, forward  # noqa: E402

P0 = example(); rng = np.random.default_rng(0)
g = th.Generator(device="cuda").manual_seed(5)
X = th.randn(2_000_000, 10, device="cuda", generator=g, dtype=th.float64)
o = X.argsort(-1, descending=True); few = (X > 0).sum(-1) <= 1
print("noise copy | measured: error B C D | estimate: total e23 e12 | estimate minus measured")
for sigma in (0.0, 0.005, 0.01, 0.02):
    for c in range(3 if sigma else 1):
        P = tuple(w * (1 + sigma * rng.standard_normal(w.shape)) for w in P0)
        ans = forward(P, X).argmax(-1); wrong = ans != o[:, 1]
        k = lambda t: 100 * t.double().mean().item()
        B = k(wrong & ~few & (ans == o[:, 2])); C = k(wrong & ~few & (ans == o[:, 0])); D = k(wrong) - k(wrong & few) - B - C
        tot, cost, e23, e12 = estimate(P)
        print(f"{sigma:.3f} {c} | {k(wrong):5.2f} {B:5.2f} {C:5.2f} {D:5.2f} | {100*tot:5.2f} {100*e23:5.2f} {100*e12:5.2f} | {100*tot - k(wrong):+5.2f}")

"""Bisection estimate (v3) and exact estimate (v4) on the same noisy copies of ARC's network (draws as noisy_v4.py)."""
import itertools
import sys
from pathlib import Path

import numpy as np
import torch as th

sys.path.insert(0, str(Path(__file__).parent))
import bias_formula_v4 as v4  # noqa: E402
from pl_flip import bisect  # noqa: E402

from algzoo.core import example, forward  # noqa: E402

pl, N, r = v4.pl, v4.N, v4.r


def estimate_v3(P):
    e23 = 0.0
    for g, w in zip(v4.GRID, v4.WTS):
        base = np.full((N, 10), -0.46); base[r, pl[:, 0]] = g; base[r, pl[:, 1]] = 1.0
        b = bisect(P, base, pl[:, 2], pl[:, 1], pl[:, 2], 0.3, min(1.7, g - 1e-3)) - 1.0
        e23 += w * np.mean(v4.F(v4.F23, -b))
    base = np.full((N, 10), -0.46); base[r, pl[:, 0]] = 1.6; base[r, pl[:, 2]] = 0.5
    b12 = (bisect(P, base, pl[:, 1], pl[:, 1], pl[:, 0], 0.6, 2.6) - 1.6) / 1.6
    return 11 / 1024 * 8 / 9 + e23 + np.mean(v4.F(v4.F12, -b12))


P0 = example(); rng = np.random.default_rng(0)
g = th.Generator(device="cuda").manual_seed(5)
X = th.randn(2_000_000, 10, device="cuda", generator=g, dtype=th.float64)
o = X.argsort(-1, descending=True)
rows = {}
for sigma in (0.0, 0.005, 0.01, 0.02):
    for c in range(3 if sigma else 1):
        P = tuple(w * (1 + sigma * rng.standard_normal(w.shape)) for w in P0)
        err = 100 * (forward(P, X).argmax(-1) != o[:, 1]).double().mean().item()
        a3, a4 = 100 * estimate_v3(P), 100 * v4.estimate(P)[0]
        rows.setdefault(sigma, []).append((err - a3, err - a4))
        print(f"noise {sigma:.3f} copy {c}: error {err:5.2f}; bisection estimate {a3:5.2f} (miss {a3-err:+.2f}); exact {a4:5.2f} (miss {a4-err:+.2f})")
for s, v in rows.items():
    v = np.array(v); print(f"noise {s:.3f}: root-mean-square miss bisection {np.sqrt(np.mean(v[:,0]**2)):.2f}, exact {np.sqrt(np.mean(v[:,1]**2)):.2f}")

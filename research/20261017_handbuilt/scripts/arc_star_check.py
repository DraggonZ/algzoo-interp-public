"""Why does evening out ties (ARC*) not change accuracy? For ARC and each ARC* file: mistakes by kind on 2 million
lists, the exact estimate (bias_formula_v4), and at the 720 filler ties (g = 1.6) the margin and the vote slope."""
import itertools
import sys
from pathlib import Path

import numpy as np
import torch as th

sys.path.insert(0, str(Path(__file__).parent))
from bias_formula_v4 import estimate  # noqa: E402
from bias_anatomy import anatomy  # noqa: E402

from algzoo.core import example, forward  # noqa: E402

pl = np.array(list(itertools.permutations(range(10), 3))); r = np.arange(len(pl))
g = th.Generator(device="cuda").manual_seed(5)
X = th.randn(2_000_000, 10, device="cuda", generator=g, dtype=th.float64)
o = X.argsort(-1, descending=True); few = (X > 0).sum(-1) <= 1
nets = {"ARC": example()}
for f in sys.argv[1:]:
    nets[f] = tuple(w.numpy() for w in th.load(f))
for name, P in nets.items():
    ans = forward(P, X).argmax(-1); wrong = ans != o[:, 1]
    k = lambda t: 100 * t.double().mean().item()
    B = k(wrong & ~few & (ans == o[:, 2])); C = k(wrong & ~few & (ans == o[:, 0])); A = k(wrong & few)
    est, _, e23, e12 = estimate(P)
    m, _, _, _, c = anatomy(P, 1.6); c3 = c[r, pl[:, 2]]
    print(f"{name}: error {k(wrong):.2f} (A {A:.2f}, B {B:.2f}, C {C:.2f}, other {k(wrong)-A-B-C:.2f}); exact estimate {100*est:.2f} (e23 {100*e23:.2f}, e12 {100*e12:.2f}); "
          f"filler ties: margin rms {np.sqrt(np.mean(m**2)):.2f}, slope median {np.median(c3):.1f}, median |margin/slope| {np.median(np.abs(m/c3)):.4f}")

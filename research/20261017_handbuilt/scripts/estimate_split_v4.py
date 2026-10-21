"""Split each network's estimate error by kind of mistake, for the exact estimate (bias_formula_v4).
Kinds: A at most one positive number; B answer is the third-largest; C answer is the largest; D anything else.
Estimate terms: A = 11/1024 x 8/9, B = e23, C = e12, D = 0. Measured on 4 million random lists per network."""
import json
import sys
from pathlib import Path

import numpy as np
import torch as th

sys.path.insert(0, str(Path(__file__).parent))
from bias_formula_v4 import estimate as est4  # noqa: E402

from algzoo.core import example, forward, load, zoo  # noqa: E402


def estimate(P):
    _, _, e23, e12 = est4(P)
    return 11 / 1024 * 8 / 9, e23, e12


def measured(P, n=4_000_000, seed=7):
    g = th.Generator(device="cuda").manual_seed(seed); out = np.zeros(4)
    for _ in range(n // 1_000_000):
        xx = th.randn(1_000_000, 10, device="cuda", generator=g, dtype=th.float64)
        ans = forward(P, xx).argmax(-1); o = xx.argsort(-1, descending=True)
        wrong = ans != o[:, 1]; few = (xx > 0).sum(-1) <= 1
        A = wrong & few; B = wrong & ~few & (ans == o[:, 2]); C = wrong & ~few & (ans == o[:, 0]); Dd = wrong & ~few & ~(ans == o[:, 2]) & ~(ans == o[:, 0])
        out += np.array([t.double().mean().item() for t in (A, B, C, Dd)])
    return out / (n // 1_000_000)


models = {"ARC": example(), **{f"zoo{s}": zoo(16, 10, s) for s in range(5)}}
for f in sorted(Path("/data/output/research/algzoo/20261016_run3/models").glob("2nd_argmax_16_10_*.pth")):
    models["s" + f.stem.split("_")[-1]] = load(str(f))
rows = {}
print("network | measured A B C D (pts) | estimate A B C (pts) | estimate minus measured: A B C D")
for k, P in models.items():
    m = 100 * measured(P); e = 100 * np.array([*estimate(P), 0.0])
    rows[k] = {"measured": m.tolist(), "estimate": e.tolist()}
    print(f"{k:6s} | {m[0]:.2f} {m[1]:.2f} {m[2]:.2f} {m[3]:.2f} | {e[0]:.2f} {e[1]:.2f} {e[2]:.2f} | {e[0]-m[0]:+.2f} {e[1]-m[1]:+.2f} {e[2]-m[2]:+.2f} {-m[3]:+.2f}")
json.dump(rows, open("estimate_split_v4.json", "w"), indent=1)
for lab, ks in (("22 development", [k for k in rows if not k.startswith("s2")]), ("16 later", [k for k in rows if k.startswith("s2")])):
    d = np.array([np.array(rows[k]["estimate"]) - np.array(rows[k]["measured"]) for k in ks])
    print(f"{lab}: mean estimate minus measured per kind A {d[:,0].mean():+.2f}, B {d[:,1].mean():+.2f}, C {d[:,2].mean():+.2f}, D {d[:,3].mean():+.2f}; "
          f"root-mean-square per kind {np.sqrt((d**2).mean(0)).round(2).tolist()}; total rms {np.sqrt((d.sum(1)**2).mean()):.2f}")

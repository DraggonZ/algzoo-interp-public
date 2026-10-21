"""bias_formula_v3 made exact. Bisection assumed one crossing, but near a tie the margin zigzags (several crossings
within 0.02). Here pl_flip.walk follows the line one linear piece at a time and wrong_mass integrates the gap
density over every stretch where the wrong number wins. Cost: 2 run-equivalents per piece (forward plus tangent).
Writes bias_formula_v4.json: {network: [measured error, predicted error, run-equivalents]}."""
import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from pl_flip import walk, wrong_mass  # noqa: E402

from algzoo.core import example, load, zoo  # noqa: E402

pl = np.array(list(itertools.permutations(range(10), 3))); N = len(pl); r = np.arange(N)
GRID = [1.05, 1.15, 1.3, 1.6, 2.0, 3.0, 5.0]
rng = np.random.default_rng(1)
x = np.sort(rng.standard_normal((5_000_000, 10)), 1)
m1, m2, m3 = x[:, -1], x[:, -2], x[:, -3]
pos = m2 > 0
rel23, rel12 = np.where(pos, (m2 - m3) / np.abs(m2), np.inf), np.where(pos, (m1 - m2) / m1, np.inf)
ratio = (m1 / m2)[pos & (rel23 < 0.01)]
edges = np.r_[1.0, [(a + b) / 2 for a, b in zip(GRID[:-1], GRID[1:])], np.inf]
WTS = np.array([np.mean((ratio >= a) & (ratio < b)) for a, b in zip(edges[:-1], edges[1:])])
D = np.linspace(0, 1, 2001)
F23 = np.array([np.mean(rel23 < d) for d in D]); F12 = np.array([np.mean(rel12 < d) for d in D])
F = lambda tab, b: np.interp(np.clip(b, 0, 1), D, tab)


def estimate(P, lo23=0.3, lo12=0.6):
    e23, cost = 0.0, 0
    f23 = lambda d: np.interp(np.clip(d, 0, 1), D, F23); f12 = lambda d: np.interp(np.clip(d, 0, 1), D, F12)
    for g, w in zip(GRID, WTS):
        base = np.full((N, 10), -0.46); base[r, pl[:, 0]] = g; base[r, pl[:, 1]] = 1.0
        pieces, ps = walk(P, base, pl[:, 2], pl[:, 1], pl[:, 2], 1.0, lo23)
        e23 += w * np.mean([wrong_mass(pc, f23) for pc in pieces]); cost += 2 * ps.sum()
    base = np.full((N, 10), -0.46); base[r, pl[:, 0]] = 1.6; base[r, pl[:, 2]] = 0.5
    pieces, ps = walk(P, base, pl[:, 1], pl[:, 1], pl[:, 0], 1.6, lo12)
    e12 = np.mean([wrong_mass(pc, f12, scale=1.6, s_ref=1.6) for pc in pieces]); cost += 2 * ps.sum()
    return 11 / 1024 * 8 / 9 + e23 + e12, cost, e23, e12


if __name__ == "__main__":
    models = {"ARC": example(), **{f"zoo{s}": zoo(16, 10, s) for s in range(5)}}
    for f in sorted(Path("/data/output/research/algzoo/20261016_run3/models").glob("2nd_argmax_16_10_*.pth")):
        models["s" + f.stem.split("_")[-1]] = load(str(f))
    v3 = json.load(open("bias_formula_v3.json"))
    rows = {}
    for k, P in models.items():
        pred, cost, _, _ = estimate(P)
        rows[k] = (v3[k][0], pred, int(cost))
    json.dump(rows, open("bias_formula_v4.json", "w"), indent=1)
    for lab, ks in (("22 development", [k for k in rows if not k.startswith("s2")]), ("16 later", [k for k in rows if k.startswith("s2")])):
        e = np.array([rows[k][0] for k in ks]); p = np.array([rows[k][1] for k in ks]); p3 = np.array([v3[k][1] for k in ks])
        c = np.array([rows[k][2] for k in ks])
        print(f"{lab}: miss {100*np.sqrt(np.mean((p-e)**2)):.2f} pts (bisection, v3, {100*np.sqrt(np.mean((p3-e)**2)):.2f}); "
              f"largest change in a prediction {100*np.max(np.abs(p-p3)):.3f} pts; run-equivalents per network mean {c.mean():.0f} (bisection 138,240)")

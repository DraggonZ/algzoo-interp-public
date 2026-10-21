"""Exact two-spike accuracy using time-shift invariance: 18 basic runs instead of 90.

With no constant terms and zero inputs before the first spike, the state is zero until the first spike. So the pair
(first spike at s, second at s + gap) is the basic pair (0, gap) run for 10 - s steps. One run per (gap, order) gives
the cells after every step; each step t >= gap is the final state of the pair whose first spike is at 9 - t.
Checked against exact_z2 (same angle cells and densities), and multiply-adds compared.
"""
import sys
import time

import numpy as np

from algzoo.core import example, zoo
from algzoo.exact_z2 import LO, HI, N_SEQ, angle_mass, correct_intervals, crossings, exact_z2


def shift_z2(params):
    w_hi, w_hh, w_oh = params
    d = w_hh.shape[0]
    ops = [0]
    total = 0.0
    for gap in range(1, N_SEQ):
        for max_first in (True, False):
            cells = [(LO, HI, np.zeros((d, 2)))]
            for t in range(N_SEQ):
                e = np.zeros(2)
                if t == 0:
                    e[0 if max_first else 1] = 1.0
                elif t == gap:
                    e[1 if max_first else 0] = 1.0
                new = []
                for a, b, A in cells:
                    B = w_hh @ A + np.outer(w_hi, e)
                    ops[0] += d * d * 2 if t > 0 else 0
                    cuts = sorted({c for j in range(d) for c in crossings(B[j], a, b)})
                    ops[0] += d * 4
                    edges = [a] + cuts + [b]
                    for lo, hi in zip(edges[:-1], edges[1:]):
                        mid = 0.5 * (lo + hi)
                        act = B @ np.array([np.cos(mid), np.sin(mid)]) > 0
                        new.append((lo, hi, B * act[:, None]))
                        ops[0] += d * 2
                cells = new
                if t >= gap:
                    s = N_SEQ - 1 - t                    # first spike position of the real pair
                    p1, p2 = (s, s + gap) if max_first else (s + gap, s)
                    good = correct_intervals(params, p1, p2, cells, ops)
                    total += sum(angle_mass(lo, hi) for lo, hi in good)
    return total, ops[0]


if __name__ == "__main__":
    for name in ["example", 0, 1, 2, 3, 4]:
        p = example() if name == "example" else zoo(16, 10, name)
        t0 = time.time(); z, ops = shift_z2(p); t1 = time.time()
        ze, opse, _ = exact_z2(p); t2 = time.time()
        print(f"{name}: shift version {z:.8f} ({ops/2880:.0f} samples' worth, {t1-t0:.1f}s) | exact_z2 {ze:.8f} ({opse/2880:.0f} samples' worth, {t2-t1:.1f}s) | diff {z-ze:+.1e}")
        sys.stdout.flush()

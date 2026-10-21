"""Do tie margins have the form m(p1, p2, p3) = f(p1, p3) - f(p1, p2), one effective vote gain per position given
where the largest is? (At a tie the list is the same when p2 and p3 swap, so m is exactly antisymmetric.)
For each p1, fit f(p1, .) to the 72 ordered pairs by least squares and report how much of the margins it explains.
Also the slope c3 (vote per unit) so margins can be read as biases (m / c3)."""
import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from bias_anatomy import anatomy  # noqa: E402

from algzoo.core import example, load, zoo  # noqa: E402

pl = np.array(list(itertools.permutations(range(10), 3))); r = np.arange(len(pl))
models = {"ARC": example(), **{f"zoo{s}": zoo(16, 10, s) for s in range(5)}}
for f in sorted(Path("/data/output/research/algzoo/20261016_run3/models").glob("2nd_argmax_16_10_*_1*.pth"))[:6]:
    models["s" + f.stem.split("_")[-1]] = load(str(f))
for g in (1.6,):
    for name, P in models.items():
        m, _, _, _, c = anatomy(P, g)
        b = m                                                   # margin (dividing by the slope blows up where it is near zero)
        fit = np.zeros_like(m)
        for p1 in range(10):
            sel = pl[:, 0] == p1; others = [q for q in range(10) if q != p1]; col = {q: j for j, q in enumerate(others)}
            A = np.zeros((sel.sum(), 9))
            for i, (_, p2, p3) in enumerate(pl[sel]):
                A[i, col[p3]] += 1; A[i, col[p2]] -= 1
            coef, *_ = np.linalg.lstsq(A, b[sel], rcond=None); fit[sel] = A @ coef
        before = pl[:, 1] < pl[:, 0]
        r2 = lambda y, yh: 1 - np.sum((y - yh) ** 2) / np.sum(y ** 2)
        print(f"{name:5s} g={g}: margin rms {np.sqrt(np.mean(b**2)):.2f}; explained by per-position gains: all {r2(b, fit):.2f}, "
              f"second before largest {r2(b[before], fit[before]):.2f}, after {r2(b[~before], fit[~before]):.2f}")

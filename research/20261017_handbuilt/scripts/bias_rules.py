"""Do two rules explain ARC's 720 tie biases? (1) Code mismatch: in mixed pairs, the number that arrived before the
largest (revealed code) is favoured. (2) Recency: within a code, the later number is favoured. Bias = m / c3 at the
filler ties (bias_anatomy, g = 1.6), clipped to [-0.1, 0.1]; positive means the third is favoured. Fit a small
linear model on rule features and report the share of variance explained, against per-position gains (a third)."""
import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from bias_anatomy import anatomy  # noqa: E402

from algzoo.core import example, zoo  # noqa: E402

pl = np.array(list(itertools.permutations(range(10), 3))); r = np.arange(len(pl))
p1, p2, p3 = pl.T
for name, P in (("ARC", example()), ("zoo0", zoo(16, 10, 0)), ("zoo2", zoo(16, 10, 2))):
    for g in (1.3, 1.6, 3.0):
        m, _, _, _, c = anatomy(P, g); b = np.clip(m / c[r, p3], -0.1, 0.1)
        b3, b2 = p3 < p1, p2 < p1                               # third / second arrived before the largest
        same = b3 == b2
        later3 = (p3 > p2).astype(float) * 2 - 1                 # +1 if the third arrived later
        feats = {"code mismatch": np.where(same, 0, np.where(b3, 1, -1)).astype(float),
                 "recency, both before": np.where(same & b3, later3, 0), "recency, both after": np.where(same & ~b3, later3, 0),
                 "third is last input": (p3 == 9).astype(float) - (p2 == 9).astype(float)}
        A = np.column_stack(list(feats.values()) + [np.ones(len(b))])
        coef, *_ = np.linalg.lstsq(A, b, rcond=None); fit = A @ coef
        r2 = 1 - np.sum((b - fit) ** 2) / np.sum((b - b.mean()) ** 2)
        print(f"{name} g={g}: variance explained by the rules {r2:.2f}; coefficients " + ", ".join(f"{k} {v:+.4f}" for k, v in zip(feats, coef)))

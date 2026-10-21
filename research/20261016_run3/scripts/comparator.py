"""Where is the second-third comparison made? (deterministic, ARC's model by default)

Largest m1 = 1.4 at p1, stored second m = 0.9 at p2, newcomer x at p3 (arriving after both), others at -0.46.
Along x in [m - 0.1, m + 0.1]: the final margin score[p3] - score[p2], its zero x*, and every on/off switch (neuron, step)
with the x where it happens. Prints a few placements, then summary statistics over all placements with p3 last.
"""
import itertools
import sys

import numpy as np

from algzoo.core import example, zoo

name = sys.argv[1] if len(sys.argv) > 1 else "example"
P = tuple(np.asarray(q, np.float64) for q in (example() if name == "example" else zoo(16, 10, int(name))))
w_hi, w_hh, w_oh = P
b, m1, m = -0.46, 1.4, 0.9
xs = m + np.linspace(-0.1, 0.1, 2001)


def trace(p1, p2, p3):
    X = np.full((len(xs), 10), b); X[:, p1] = m1; X[:, p2] = m; X[:, p3] = xs
    h = np.zeros((len(xs), 16)); on = []
    for t in range(10):
        h = np.maximum(0, h @ w_hh.T + X[:, t, None] * w_hi); on.append(h > 0)
    s = h @ w_oh.T
    return s[:, p3] - s[:, p2], s.argmax(1), np.stack(on, 1)      # margin, pick, [x, step, neuron]


rows = []
cfg = [c for c in itertools.permutations(range(10), 3) if c[2] > max(c[0], c[1])]
for k, (p1, p2, p3) in enumerate(cfg):
    M, pick, on = trace(p1, p2, p3)
    flips = np.where(np.diff(np.sign(M)) != 0)[0]
    xstar = xs[flips[0]] if len(flips) else np.nan
    sw = np.argwhere(on[1:] != on[:-1])                            # (x index, step, neuron)
    events = [(xs[i], t, j) for i, t, j in sw]
    near = [e for e in events if abs(e[0] - xstar) < 0.02] if len(flips) else []
    slope_lo = (M[900] - M[800]) / (xs[900] - xs[800]); slope_hi = (M[1200] - M[1100]) / (xs[1200] - xs[1100])
    rows.append((p1, p2, p3, xstar - m, len(flips), len(events), slope_lo, slope_hi,
                 min((abs(e[0] - xstar) for e in events), default=np.nan)))
    if k in (0, 40, 80, 100):
        print(f"placement largest {p1}, second {p2}, newcomer {p3}: crossing at x - m = {xstar - m:+.4f}; margin slope below {slope_lo:.0f}, above {slope_hi:.0f}; "
              f"{len(events)} switches in the window; nearest to the crossing: " + ", ".join(f"n{j} step {t} at {e - m:+.4f}" for e, t, j in sorted(near, key=lambda z: abs(z[0] - xstar))[:4]))
R = np.array(rows, dtype=float)
ok = ~np.isnan(R[:, 3])
print(f"{name}: {len(R)} placements with the newcomer last; crossing found in {ok.mean():.2f}; |x* - m| median {np.median(np.abs(R[ok, 3])):.4f}, 90th pct {np.percentile(np.abs(R[ok, 3]), 90):.4f}; "
      f"more than one crossing {np.mean(R[ok, 4] > 1):.2f}; switches in window median {np.median(R[:, 5]):.0f}; "
      f"slope below median {np.median(R[:, 6]):.0f}, above {np.median(R[:, 7]):.0f}; distance from crossing to nearest switch median {np.nanmedian(R[ok, 8]):.4f}")

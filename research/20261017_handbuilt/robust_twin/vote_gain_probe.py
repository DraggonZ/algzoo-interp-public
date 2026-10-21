"""Mechanism probe (not a search): in v18 at trained scale (g = a = w = 10, lam 10, tie-breaker 1% of a vote), raise the
vote gain A0 = 1 - c from 10 to 20 and 40 while the timer level stays g = 10. Prediction: the lead grows with A0, so
the timer's sensitivity to noise falls, but the timer clips more often (its level g (m1 - x) must exceed a wave of
size about A d), so accuracy falls unless the rules handle clipping. Measures: accuracy, median lead, timer clipping
rate (steps where some but not all of clock0-2 and flip are off, among steps with a live timer), sensitivity of the
flip and clock neurons (as sensitivity.py), accuracy under 1% noise."""
import numpy as np
from algzoo.handbuilt_v18 import build_v18
import simlib
from sensitivity import sens

X = simlib.lists(72, 20_000); y = X.argsort(1)[:, -2]; rng = np.random.default_rng(0); age = simlib.task_vars(X)["age"]
for A0 in (10, 20, 40):
    P = tuple(np.asarray(p, float) for p in build_v18(g=10, a=10, w=10, lam=10.0, c=1 - A0, eps=0.01 * A0).params)
    L, Pre, H = simlib.run(X, params=P); k = L.argmax(1); j = np.argsort(L, 1)[:, -2]; lead = L[np.arange(len(X)), k] - L[np.arange(len(X)), j]
    off = (Pre[:, :, [3, 4, 5, 6]] <= 0).sum(-1); live = age >= 1; clip = ((off > 0) & (off < 4))[live].mean()
    s, typ = sens(P, X)
    h = np.zeros((len(X), 16))
    for t in range(10):
        h = np.maximum(h @ P[1].T + X[:, t:t + 1] * P[0], 0); h = np.maximum(h + 0.01 * typ * rng.standard_normal(h.shape) * (h > 0), 0)
    print(f"vote gain {A0:3d}: accuracy {(k == y).mean():.3f}; median lead {np.median(lead):.2f}; timer partly clipped on {clip:.3f} of live steps; "
          f"sensitivity flip {s[3]:.2f}, clocks {s[4]:.2f} {s[5]:.2f} {s[6]:.2f}, store {s[0]:.2f}; accuracy under 1% noise {((h @ P[2].T).argmax(1) == y).mean():.3f}", flush=True)

"""Relative votes, sizes measured from ARC (vote_profiles.log, fillers -0.5). A kept non-record x at position p votes
k[p] * min(x - 0.7 * second-so-far, cap)^+ on its own position; an old largest P at p votes r[p] * P when beaten;
the last input votes like a kept number. Nothing else. Compare with ARC."""
import numpy as np, torch as th
from algzoo.core import example, forward

rng = np.random.default_rng(7)
x = rng.standard_normal((1_000_000, 10)); B, n = x.shape
z = np.zeros((B, 2))
m2p = np.stack([np.sort(np.c_[z, x[:, :t]], 1)[:, -2] for t in range(n)], 1)
m1p = np.stack([np.max(np.c_[z, x[:, :t]], 1) for t in range(n)], 1)
record = x > m1p; final = x == x.max(1, keepdims=True)
tgt = x.argsort(1)[:, -2]; third = x.argsort(1)[:, -3]; amax = x.argmax(1)
arc = forward(example(), th.tensor(x)).numpy().argmax(1)
srt = np.sort(x, 1); gap = (srt[:, -2] - srt[:, -3]) / np.abs(srt[:, -2]).clip(1e-9)

k_meas = np.array([30, 30, 30, 36, 39, 70, 71, 62, 67, 67.0])     # positions 0, 1 and 9 not measured: copied
r_meas = np.array([25, 25, 28, 34, 37, 53, 63, 54, 54, 54.0])


def model(k, r, c=0.7, cap=1.0):
    kept_v = np.minimum(x - c * m2p, cap).clip(0) * k
    v = np.where(record, 0.0, kept_v)
    v = np.where(record & ~final, r * x, v)
    return v.argmax(1)


def report(P, name):
    m = gap < 0.03
    cls = {"before": (tgt < amax) & (third < amax), "after": (tgt > amax) & (third > amax)}
    cls["mixed"] = ~(cls["before"] | cls["after"])
    s = " ".join(f"{c} {1 - (P[mm & m] == tgt[mm & m]).mean():.2f}" for c, mm in cls.items())
    w = arc != tgt
    print(f"{name:40s} acc {(P == tgt).mean():.4f} agree {(P == arc).mean():.3f} ARC-mistakes {(P[w] == arc[w]).mean():.2f} | near-tie error {s}")


report(arc, "ARC")
report(model(np.ones(10) * 70, np.ones(10) * 70, c=0.0, cap=9), "equal sizes, no discount (fair)")
report(model(k_meas, r_meas, c=0.0, cap=9), "measured fading, no discount")
report(model(k_meas, r_meas), "measured fading, discount 0.7, cap 1")
report(model(k_meas, r_meas, cap=9), "measured fading, discount 0.7, no cap")
for c in (0.5, 0.6, 0.8):
    report(model(k_meas, r_meas, c=c), f"measured fading, discount {c}, cap 1")

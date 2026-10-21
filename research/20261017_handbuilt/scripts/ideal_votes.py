"""Idealized vote model: which kernel and vote rule can reach >90%?

Votes: every number that is not the current max casts a vote equal to its value,
except numbers that arrived before the latest record and were not records
themselves (their votes sit in the wipeable copy and are erased). Score of slot k
= sum_i vote_i * K(age_i - age_k). K comes from a clock: sum of cosines.
"""
import numpy as np

rng = np.random.default_rng(0)
x = rng.standard_normal((200_000, 10))
n = 10
tgt = x.argsort(1)[:, -2]
pos = np.arange(n)
m = np.maximum.accumulate(x, 1)
prevmax = np.concatenate([np.full((len(x), 1), -np.inf), m[:, :-1]], 1)
rec = x > prevmax
last_rec = np.where(rec, pos, -1).max(1)
amax = x.argmax(1)


def votes(rectify, records_only_if_positive=True):
    v = x.copy()
    if rectify:
        v = np.maximum(v, 0)
    keep = (pos[None] > last_rec[:, None]) | rec  # after last record, or a record
    v = np.where(keep, v, 0.0)
    v[np.arange(len(x)), amax] = 0.0  # max hidden
    return v


def kernel(freqs, weights, omega):
    d = pos[:, None] - pos[None]
    return sum(w * np.cos(f * omega * d) for f, w in zip(freqs, weights))


for rectify in (False, True):
    v = votes(rectify)
    print("rectified" if rectify else "raw x", " perfect-kernel acc", (v.argmax(1) == tgt).mean().round(4))
    for name, fr, wt, om in [("1 freq 36deg", [1], [1], np.pi / 5), ("1 freq 30deg", [1], [1], np.pi / 6),
                             ("1 freq 42deg", [1], [1], 42 / 180 * np.pi),
                             ("2 freq 36,72", [1, 2], [1, 1], np.pi / 5), ("2 freq 18,36", [1, 2], [1, 1], np.pi / 10),
                             ("3 freq 36", [1, 2, 3], [1, 1, 1], np.pi / 5), ("4 freq 36", [1, 2, 3, 4], [1, 1, 1, 1], np.pi / 5)]:
        s = v @ kernel(fr, wt, om)
        print(f"   {name:14s} acc {(s.argmax(1) == tgt).mean():.4f}")

# Kernel shaped like ARC's measured one: neighbours ~0.4, two away ~0, further ~ -0.2
print("\nkernels shaped by offset (K0=1):")
for name, prof in [("trained-like", [1, .4, 0, -.25, -.25, -.15, -.1, -.1, -.1, -.1]),
                   ("nbr .4, rest 0", [1, .4, 0, 0, 0, 0, 0, 0, 0, 0]),
                   ("nbr .6, rest 0", [1, .6, 0, 0, 0, 0, 0, 0, 0, 0]),
                   ("nbr .8, rest 0", [1, .8, 0, 0, 0, 0, 0, 0, 0, 0]),
                   ("delta", [1, 0, 0, 0, 0, 0, 0, 0, 0, 0])]:
    K = np.array(prof)[np.abs(pos[:, None] - pos[None])]
    for rectify in (False, True):
        s = votes(rectify) @ K
        print(f"   {name:15s} {'rect' if rectify else 'raw '} acc {(s.argmax(1) == tgt).mean():.4f}")

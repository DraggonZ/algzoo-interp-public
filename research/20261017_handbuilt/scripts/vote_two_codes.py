"""Simplified model with two codes, as a 16-neuron budget forces. Beaten records (old largests) vote in a broad
clock code: value * lam * cos(40 deg * distance) over the 9-cycle of slots (one shared plane, so several such votes
add up as vectors). Kept non-records and the last input vote sharply on their own slot, with the trained cut rule.
Compare accuracy and where mistakes fall with ARC's."""
import numpy as np, torch as th
from algzoo.core import example, forward

rng = np.random.default_rng(6)
x = rng.standard_normal((1_000_000, 10)); B, n = x.shape
z = np.zeros((B, 2))
m2p = np.stack([np.sort(np.c_[z, x[:, :t]], 1)[:, -2] for t in range(n)], 1)
m1p = np.stack([np.max(np.c_[z, x[:, :t]], 1) for t in range(n)], 1)
lb = np.stack([(x[:, t + 1:] > x[:, t:t + 1]).sum(1) for t in range(n)], 1)
record = x > m1p
final = x == x.max(1, keepdims=True)
tgt = x.argsort(1)[:, -2]; amax = x.argmax(1)
arc = forward(example(), th.tensor(x)).numpy().argmax(1)
srt = np.sort(x, 1); gap = (srt[:, -2] - srt[:, -3]) / np.abs(srt[:, -2]).clip(1e-9)
third = x.argsort(1)[:, -3]
ang = np.deg2rad(40.0) * np.arange(n)                       # slot angle (position * 40 deg)
C = np.cos(ang[:, None] - ang[None, :])                      # C[j, k]: clock vote at j read at k


def model(lam=1.0, kept_c=0.75, kept=(1, .6, .2, -.5), broad_rec=True):
    k = record | ((x >= kept_c * m2p) & (x > 0))
    L = lb.clip(max=3)
    sharp = np.where(record, 0.0, np.where(k, np.take(kept, L), 0.0)) * x
    sharp[:, -1] = np.where(final[:, -1], 0, x[:, -1])
    rec = np.where(record & ~final, x, 0.0)
    if broad_rec:
        s = sharp + lam * rec @ C
    else:
        s = sharp + rec
    return s.argmax(1)


def profile(P, name):
    acc = (P == tgt).mean()
    m = gap < 0.03
    cls = {"both before largest": (tgt < amax) & (third < amax), "both after": (tgt > amax) & (third > amax)}
    cls["mixed"] = ~(cls["both before largest"] | cls["both after"])
    s = "  ".join(f"{c} {1 - (P[mm & m] == tgt[mm & m]).mean():.2f}" for c, mm in cls.items())
    print(f"{name:36s} acc {acc:.4f}  agree with ARC {(P == arc).mean():.3f}  error at gaps < 3%: {s}")


profile(arc, "ARC")
profile(model(broad_rec=False), "sharp everywhere")
for lam in (0.8, 1.0, 1.2, 1.5, 2.0):
    profile(model(lam=lam), f"old largest broad, lam {lam}")

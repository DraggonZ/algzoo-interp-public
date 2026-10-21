"""Hand-written vote rules, no fitting. Each number votes value * size on its own position (plus optional spill to
neighbours); highest total wins. Sizes depend on the number's kind and on how many later numbers beat it."""
import numpy as np

rng = np.random.default_rng(5)
x = rng.standard_normal((1_000_000, 10)); B, n = x.shape
z = np.zeros((B, 2))
m2p = np.stack([np.sort(np.c_[z, x[:, :t]], 1)[:, -2] for t in range(n)], 1)
m1p = np.stack([np.max(np.c_[z, x[:, :t]], 1) for t in range(n)], 1)
lb = np.stack([(x[:, t + 1:] > x[:, t:t + 1]).sum(1) for t in range(n)], 1)
record = x > m1p
final = x == x.max(1, keepdims=True)
tgt = x.argsort(1)[:, -2]


def score(kept_c=0.75, rec=(0, 1, 1, 1), kept=(1, 1, 1, 1), forgot=0.0, last=None, spill=0.0):
    """rec[i], kept[i]: vote size for an old largest / kept non-record with i later bigger numbers (3 = 3+)."""
    k = record | ((x >= kept_c * m2p) & (x > 0))
    L = lb.clip(max=3)
    size = np.where(record, np.take(rec, L), np.where(k, np.take(kept, L), forgot))
    size[final] = 0
    if last is not None: size[:, -1] = np.where(final[:, -1], 0, last)
    v = size * x
    s = v.copy()
    if spill:
        s[:, 1:] += spill * v[:, :-1]; s[:, :-1] += spill * v[:, 1:]
    return (s.argmax(1) == tgt).mean()


rows = [
    ("all kept votes equal, no shrink (delta kernel)", dict()),
    ("flat 25% shrink per later bigger, all kinds", dict(rec=(1, 1, .75, .56), kept=(1, .75, .56, .42))),
    ("  same, plus 40% spill", dict(rec=(1, 1, .75, .56), kept=(1, .75, .56, .42), spill=.4)),
    ("network-like: old largest keeps vote, kept cut by later bigger", dict(rec=(0, 1, .85, .5), kept=(1, .6, .2, -.5))),
    ("  plus forgotten numbers vote 0.4x", dict(rec=(0, 1, .85, .5), kept=(1, .6, .2, -.5), forgot=.4)),
    ("  plus last input votes 0.7x", dict(rec=(0, 1, .85, .5), kept=(1, .6, .2, -.5), forgot=.4, last=.7)),
    ("  plus 13% spill", dict(rec=(0, 1, .85, .5), kept=(1, .6, .2, -.5), forgot=.4, last=.7, spill=.13)),
    ("  plus 40% spill", dict(rec=(0, 1, .85, .5), kept=(1, .6, .2, -.5), forgot=.4, last=.7, spill=.4)),
]
for name, kw in rows:
    print(f"{name:66s} {score(**kw):.4f}")

print("\nwhat the earlier simplified model did differently")
more = [
    ("flat 25% shrink that also hits the old largest's revealing step", dict(rec=(1, .75, .56, .42), kept=(1, .75, .56, .42))),
    ("  plus last input at full size 1.0x", dict(rec=(1, .75, .56, .42), kept=(1, .75, .56, .42), last=1.0)),
    ("network-like, last input 1.0x instead of 0.7x", dict(rec=(0, 1, .85, .5), kept=(1, .6, .2, -.5), forgot=.4, last=1.0, spill=.13)),
    ("network-like, last input 0.4x", dict(rec=(0, 1, .85, .5), kept=(1, .6, .2, -.5), forgot=.4, last=.4, spill=.13)),
    ("network-like, no forgotten votes", dict(rec=(0, 1, .85, .5), kept=(1, .6, .2, -.5), last=.7, spill=.13)),
]
for name, kw in more:
    print(f"{name:66s} {score(**kw):.4f}")

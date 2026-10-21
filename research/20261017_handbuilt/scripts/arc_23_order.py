"""ARC's second-third mix-ups: does the third arrive after the second (a number
just below the second leaving a lasting vote) or before it (the second failing to
erase an earlier vote)? Same split for the hand-built v2."""
import torch as th

from algzoo import handbuilt as hb
from algzoo.core import example, forward

g = th.Generator(device="cuda").manual_seed(11)
x = th.randn(4_000_000, 10, generator=g, device="cuda", dtype=th.float64)
o = x.argsort(-1)
p1, p2, p3 = o[:, -1], o[:, -2], o[:, -3]
xs = x.sort(-1).values
tie23 = ((xs[:, -2] - xs[:, -3]) < 0.1) & ((xs[:, -1] - xs[:, -2]) >= 0.1) & (xs[:, -2] > 0)
for name, P in (("ARC", example()), ("hand v2", hb.build_v2().params)):
    pred = forward(P, x).argmax(-1)
    picks3 = tie23 & (pred == p3)
    print(name)
    for lab, m in (("third after second", p3 > p2), ("third before second", p3 < p2)):
        for lab2, mm in (("largest last of three", p1 > th.maximum(p2, p3)), ("largest earlier", p1 < th.maximum(p2, p3))):
            sel = tie23 & m & mm
            print(f"  {lab:20s} {lab2:22s}: picks third in {100*(picks3 & m & mm).sum().item()/sel.sum().item():5.1f}% of these near-ties ({100*(picks3&m&mm).float().mean().item():.2f} points)")

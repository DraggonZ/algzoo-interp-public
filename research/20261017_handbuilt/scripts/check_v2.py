"""v2: exact two-spike accuracy, GPU float32 accuracy, and mistake kinds vs ARC's model."""
import numpy as np
import torch as th

from algzoo import handbuilt as hb
from algzoo.core import accuracy, example, forward
from algzoo.exact_z2 import exact_z2

P = hb.build_v2().params
print("exact two-spike accuracy:", exact_z2(P)[0], " ARC:", exact_z2(example())[0])
for dt in (th.float32, th.float64):
    a, n = accuracy(P, n_samples=10**8 if dt == th.float32 else 10**7, dtype=dt)
    print(f"GPU accuracy {dt}: {a:.5f} (n={n})")

# mistake kinds on 4M random inputs, same table as the report (gap threshold 0.1)
g = th.Generator(device="cuda").manual_seed(7)
x = th.randn(4_000_000, 10, generator=g, device="cuda", dtype=th.float64)
xs = x.sort(-1).values
gap12, gap23 = (xs[:, -1] - xs[:, -2]), (xs[:, -2] - xs[:, -3])
onepos = xs[:, -2] < 0
tg = x.argsort(-1)[:, -2]
third = x.argsort(-1)[:, -3]
first = x.argmax(-1)
rows = [("second and third within 0.1", (gap23 < .1) & (gap12 >= .1) & ~onepos),
        ("largest and second within 0.1", (gap12 < .1) & (gap23 >= .1) & ~onepos),
        ("both gaps below 0.1", (gap12 < .1) & (gap23 < .1) & ~onepos),
        ("only one positive input", onepos),
        ("other", (gap12 >= .1) & (gap23 >= .1) & ~onepos)]
print(f"\n{'kind':32s} {'hand v2 pts':>11s} {'ARC pts':>8s}   hand picks third / largest")
for name, P_ in (("hand", P), ("ARC", example())):
    pred = forward(P_, x).argmax(-1)
    bad = pred != tg
    if name == "hand":
        hb_bad, hb_pred = bad, pred
    else:
        arc_bad = bad
for kind, m in rows:
    hbm, am = (hb_bad & m), (arc_bad & m)
    nb = hbm.sum().item()
    print(f"{kind:32s} {100*hbm.float().mean().item():11.2f} {100*am.float().mean().item():8.2f}   "
          f"{(hb_pred[hbm]==third[hbm]).float().mean().item() if nb else 0:.2f} / {(hb_pred[hbm]==first[hbm]).float().mean().item() if nb else 0:.2f}")
print("total", round(100 * hb_bad.float().mean().item(), 2), round(100 * arc_bad.float().mean().item(), 2))
# finer: within the 'other' and tie kinds, how small are the relative gaps of hand mistakes?
rel23 = gap23 / xs[:, -2].abs().clamp_min(1e-9)
rel12 = gap12 / xs[:, -1].abs()
m = hb_bad & ~onepos
print("hand mistakes (not one-positive): share with min relative gap < 1%:",
      ((th.minimum(rel12, rel23) < 0.01) & m).sum().item() / m.sum().item())

"""What kinds of mistakes do noisy copies of ARC's model make? Same categories as the report's table."""
import numpy as np
import torch as th

from algzoo.core import example, forward

g = th.Generator(device="cuda").manual_seed(9)
x = th.randn(2_000_000, 10, generator=g, device="cuda", dtype=th.float64)
xs = x.sort(-1).values; tgt = x.argsort(-1)[:, -2]
g12, g23 = xs[:, -1] - xs[:, -2], xs[:, -2] - xs[:, -3]
onepos = xs[:, -2] < 0
cats = {"2nd-3rd within 0.1": (g23 < .1) & (g12 >= .1) & ~onepos, "1st-2nd within 0.1": (g12 < .1) & (g23 >= .1) & ~onepos,
        "both within 0.1": (g12 < .1) & (g23 < .1) & ~onepos, "one positive": onepos, "other (no close call)": (g12 >= .1) & (g23 >= .1) & ~onepos}
rng = np.random.default_rng(11)
print("points of mistakes by kind: " + " | ".join(cats))
for sd in (0.0, 0.005, 0.01, 0.02):
    for i in range(3 if sd else 1):
        P = example() if sd == 0 else tuple(w * (1 + sd * rng.standard_normal(w.shape)) for w in example())
        bad = forward(P, x).argmax(-1) != tgt
        print(f"noise {sd:.1%} #{i}: total {100*bad.double().mean():5.1f} | " + " | ".join(f"{100*(bad & m).double().mean():5.2f}" for m in cats.values()))

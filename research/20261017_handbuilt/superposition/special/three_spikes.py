"""Do zero-background test inputs hide the close-call problem? Three spikes: largest 2.0, second 1.5, third r x 1.5,
at all 720 placements. Background: zeros, or the other seven drawn from a standard normal below the third
(as in a random list, by rejection). Share right (names the second's position)."""
import itertools, numpy as np, torch as th
from algzoo.core import example, forward
P = [np.asarray(p, float) for p in example()]
pl = th.tensor(list(itertools.permutations(range(10), 3)))
g = th.Generator().manual_seed(0)
def bg(n, top):                      # n x 10 normals, each below top (rejection)
    z = th.randn(n, 10, dtype=th.float64, generator=g)
    while (bad := z >= top).any(): z[bad] = th.randn(int(bad.sum()), dtype=th.float64, generator=g)
    return z
for r in (0.90, 0.95, 0.98, 0.99):
    vals = th.tensor([2.0, 1.5, 1.5 * r], dtype=th.float64)
    out = []
    for name in ("zeros", "random below third"):
        reps = 1 if name == "zeros" else 50
        X = th.zeros(len(pl) * reps, 10, dtype=th.float64) if name == "zeros" else bg(len(pl) * reps, 1.5 * r)
        p = pl.repeat(reps, 1); X[th.arange(len(X))[:, None], p] = vals
        right = (forward(P, X).argmax(-1) == p[:, 1]).double()
        out.append(f"{name}: {100 * right.mean():.1f}% right")
    print(f"third = {r:.2f} x second | " + " | ".join(out))

# Same answer as on zeros? Per placement, the share of random backgrounds giving the zero-background answer.
print("\nAgreement with the zero-background answer, per placement (50 random backgrounds each):")
for r in (0.95, 0.98, 0.99):
    vals = th.tensor([2.0, 1.5, 1.5 * r], dtype=th.float64)
    X0 = th.zeros(len(pl), 10, dtype=th.float64); X0[th.arange(len(pl))[:, None], pl] = vals
    a0 = forward(P, X0).argmax(-1)
    X = bg(len(pl) * 50, 1.5 * r); p = pl.repeat(50, 1); X[th.arange(len(X))[:, None], p] = vals
    a = forward(P, X).argmax(-1).reshape(50, len(pl))
    agree = (a == a0[None]).double().mean(0)
    print(f"third = {r:.2f} x second: mean agreement {100 * agree.mean():.1f}%; placements where the background "
          f"changes the answer at least 20% of the time: {100 * (agree < 0.8).double().mean():.0f}%")

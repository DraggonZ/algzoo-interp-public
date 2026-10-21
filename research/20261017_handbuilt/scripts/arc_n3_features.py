"""ARC: what does each neuron's final value track? R^2 of a linear fit on feature sets:
clock = largest's value x one-hot(age of largest); second = second's value x one-hot(age of
second); last = last input (and its positive part); third = third's value x one-hot(age)."""
import numpy as np
import torch as th

from algzoo.core import example, forward

g = th.Generator().manual_seed(2)
x = th.randn(300_000, 10, generator=g, dtype=th.float64)
_, H = forward(example(), x, return_hidden=True)
h = H[:, -1].numpy()
xn = x.numpy()
o = xn.argsort(1)
B = len(xn)
r = np.arange(B)


def onehot_val(rank):
    p = o[:, -rank]
    v = xn[r, p]
    F = np.zeros((B, 10))
    F[r, 9 - p] = v  # column = age
    return F


clock, second, third = onehot_val(1), onehot_val(2), onehot_val(3)
last = np.c_[xn[:, -1], np.maximum(xn[:, -1], 0)]
sets = {"clock": clock, "second": second, "third": third, "last": last,
        "clock+second": np.c_[clock, second], "clock+second+last": np.c_[clock, second, last],
        "clock+second+third+last": np.c_[clock, second, third, last]}


def r2(F, y):
    F = np.c_[F, np.ones(B)]
    c, *_ = np.linalg.lstsq(F, y, rcond=None)
    return 1 - ((F @ c - y) ** 2).mean() / y.var()


print("neuron " + " ".join(f"{k:>24s}" for k in sets))
for n in range(16):
    print(f"  {n:2d}   " + " ".join(f"{r2(F, h[:, n]):24.2f}" for F in sets.values()))

np.set_printoptions(precision=1, suppress=True, linewidth=200)
F = np.c_[clock, second, third, last, np.ones(B)]
for n in (3, 13):
    c, *_ = np.linalg.lstsq(F, h[:, n], rcond=None)
    print(f"\nneuron {n}: per unit of value, by age 0..9")
    print("  largest at age:", c[:10]); print("  second at age: ", c[10:20]); print("  third at age:  ", c[20:30])
w_oh = example()[2]
print("\noutput weights of 3 by slot (centred), slot 9 = age 0:", (w_oh - w_oh.mean(0))[:, 3].round(1))

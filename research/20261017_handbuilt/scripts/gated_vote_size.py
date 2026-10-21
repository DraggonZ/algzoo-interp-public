"""Size of the vote a later number keeps (gated code): largest 2.0 at position 1, an earlier number S at position 3,
the number x at position 6, -0.3 elsewhere. Vote = final centred score at position 6 minus the same list with x at
-0.3. Is the vote proportional to x alone, or does it depend on S? Compare with a revealed (before-largest) vote:
x at position 2, S at position 4, largest 2.0 at position 7."""
import numpy as np

from algzoo.core import example

wi, wh, wo = example()


def sc(X):
    h = np.zeros((len(X), 16))
    for t in range(10):
        h = np.maximum(0, h @ wh.T + X[:, t:t + 1] * wi)
    s = h @ wo.T
    return s - s.mean(1, keepdims=True)


for label, pL, pS, px in (("gated (x after largest)", 1, 3, 6), ("revealed (x before largest)", 7, 4, 2)):
    print(label)
    for S in (0.3, 0.6, 0.9):
        xs = np.array([0.9, 1.0, 1.1, 1.2, 1.4]); X = np.full((len(xs), 10), -0.3); X[:, pL] = 2.0; X[:, pS] = S; X[:, px] = xs
        Xn = X.copy(); Xn[:, px] = -0.3
        vote = sc(X)[:, px] - sc(Xn)[:, px]
        print(f"  S={S}: vote per unit of x at x = " + "  ".join(f"{x}: {v/x:5.1f}" for x, v in zip(xs, vote)))

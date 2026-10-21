"""What sets the trained-scale twin's small lead? Score vectors on a few random lists, and over 20,000 lists which
position is the runner-up (largest, third, a non-positive number, other) and the median lead in each case."""
import numpy as np
from algzoo.handbuilt_v18 import build_v18
import simlib

X = simlib.lists(72, 20_000); y = X.argsort(1)[:, -2]; o = X.argsort(1)[:, ::-1]
P = tuple(np.asarray(p, float) for p in build_v18(g=10, a=10, w=10, lam=10.0, c=-9.0, eps=0.1).params)
L = simlib.run(X, params=P)[0]; k = L.argmax(1); srt = np.argsort(L, 1); j = srt[:, -2]; lead = L[np.arange(len(X)), k] - L[np.arange(len(X)), j]
np.set_printoptions(precision=2, suppress=True, linewidth=160)
for i in range(4):
    print("list", X[i], "\n scores", L[i], "\n winner", k[i], "runner-up", j[i], "true second", y[i])
kind = np.select([j == o[:, 0], j == o[:, 1], j == o[:, 2], np.take_along_axis(X, j[:, None], 1)[:, 0] <= 0], ["largest", "second", "third", "non-positive"], "other")
for kk in ("largest", "second", "third", "non-positive", "other"):
    m = kind == kk; print(f"runner-up is the {kk:12s}: {m.mean():.3f} of lists; median lead {np.median(lead[m]) if m.any() else float('nan'):.2f}")

print("\ntrained network:")
L = simlib.run(X)[0]; k = L.argmax(1); srt = np.argsort(L, 1); j = srt[:, -2]; lead = L[np.arange(len(X)), k] - L[np.arange(len(X)), j]
kind = np.select([j == o[:, 0], j == o[:, 1], j == o[:, 2], np.take_along_axis(X, j[:, None], 1)[:, 0] <= 0], ["largest", "second", "third", "non-positive"], "other")
for kk in ("largest", "second", "third", "non-positive", "other"):
    m = kind == kk; print(f"runner-up is the {kk:12s}: {m.mean():.3f} of lists; median lead {np.median(lead[m]) if m.any() else float('nan'):.2f}")
lg = np.take_along_axis(L, o[:, :1], 1)[:, 0]; sc = np.take_along_axis(L, o[:, 1:2], 1)[:, 0]
print(f"median score of the largest's position {np.median(lg):.2f}; of the second's {np.median(sc):.2f}; median second minus largest {np.median(sc - lg):.2f}")

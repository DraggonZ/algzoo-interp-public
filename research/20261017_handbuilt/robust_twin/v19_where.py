"""Where does v19 (kappa 0.5, v18 gains, no clipping) go wrong that v18 does not? Wrong answers by what they name and by
list structure: whether the second-largest was a record when it arrived (an overtaken record) and whether the largest
came after it."""
import numpy as np
from algzoo.handbuilt_v19 import build_v19
import simlib

X = simlib.lists(75, 200_000); y = X.argsort(1)[:, -2]; o = X.argsort(1)[:, ::-1]; B = len(X)
cm = np.maximum.accumulate(X, 1); prev = np.c_[np.full((B, 1), -np.inf), cm[:, :-1]]; rec = X > prev
p1, p2, p3 = o[:, 0], o[:, 1], o[:, 2]; sec_rec = rec[np.arange(B), p2]
for kappa in (0.0, 0.5):
    P = tuple(np.asarray(p, float) for p in build_v19(kappa=kappa).params); a = simlib.run(X, params=P)[0].argmax(1)
    wrong = a != y
    named = np.select([a == p1, a == p3, np.take_along_axis(X, a[:, None], 1)[:, 0] <= 0], ["largest", "third", "non-positive"], "other")
    print(f"kappa {kappa}: accuracy {1 - wrong.mean():.4f}; wrong answers name: " + ", ".join(f"{k} {np.mean(named[wrong] == k):.2f}" for k in ("largest", "third", "non-positive", "other")))
    for lab, m in (("second an overtaken record", sec_rec & (p2 < p1)), ("second a kept number before the largest", ~sec_rec & (p2 < p1)), ("second after the largest", p2 > p1)):
        print(f"   {lab:40s}: {m.mean():.3f} of lists; accuracy {1 - wrong[m].mean():.3f}")
    if kappa:
        w = wrong & (named == "largest"); print(f"   when it names the largest: largest at position 9 {np.mean(p1[w] == 9):.2f}, largest at 8 {np.mean(p1[w] == 8):.2f}, second right before the largest {np.mean(p2[w] == p1[w] - 1):.2f}")
        w = wrong & (named == "third"); print(f"   when it names the third: third right after the second {np.mean(p3[w] == p2[w] + 1):.2f}, right before {np.mean(p3[w] == p2[w] - 1):.2f}")

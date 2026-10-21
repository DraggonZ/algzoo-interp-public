"""Forcing the keep gate off below 1.0 x the second-largest so far lifts accuracy (g24_forget_more.log). Which lists?
Compare the intact network with the forced one on 200,000 lists (seed 25): mistakes fixed and mistakes made, by what
the intact network named and whether that number arrived after the second-largest and below it at arrival
('late third': at its arrival it was below the second-largest so far, so it cannot be the answer)."""
import numpy as np
from gatelib import lists, run, task_vars, answer

X = lists(25); V = task_vars(X); y = answer(X); L0, P0, H0 = run(X)
rec = np.zeros((len(X), 10), bool); rec[:, 1:] = V["new"][:, :9].astype(bool)
m2 = np.zeros((len(X), 10)); m2[:, 1:] = V["m2"][:, :9]
xs = np.zeros((len(X), 10)); xs[:, 1:] = X[:, :9]
mk = rec | (xs >= m2); mk[:, 0] = True


def hook(t, hp, pre, h):
    h[:, 5] = np.maximum(pre[:, 5], 0) * mk[:, t]; return h


L1 = run(X, hook=hook)[0]
a0, a1 = L0.argmax(1), L1.argmax(1); ok0, ok1 = a0 == y, a1 == y
print(f"accuracy intact {100 * ok0.mean():.2f}%, forced off below the second-largest {100 * ok1.mean():.2f}%")
print(f"mistakes fixed {(~ok0 & ok1).sum()}, mistakes made {(ok0 & ~ok1).sum()} (of {len(X)} lists)")
o = X.argsort(1)
B = np.arange(len(X))
named_below = (X[B, a0] < V["m2"][B, a0]) & (X[B, a0] > 0)   # the named number was below the second-largest at arrival
kept_by_gate = H0[B, np.minimum(a0 + 1, 9), 5] > 0
fix = ~ok0 & ok1
print("intact mistakes, by what was named:")
for name, m in (("third-largest", a0 == o[:, -3]), ("largest", a0 == o[:, -1]), ("other", (a0 != o[:, -3]) & (a0 != o[:, -1]))):
    sel = ~ok0 & m
    print(f"  {name:14s}: {sel.sum():6d} mistakes, {100 * fix[sel].mean():5.1f}% fixed; of these mistakes the named number was below the "
          f"second-largest at arrival in {100 * named_below[sel].mean():.1f}% (gate on after it in {100 * (kept_by_gate & named_below)[sel].mean():.1f}%)")
made = ok0 & ~ok1
print(f"new mistakes: forced network names the largest in {100 * (a1[made] == o[made, -1]).mean():.1f}%, the third in {100 * (a1[made] == o[made, -3]).mean():.1f}%")
pos_named = np.bincount(a1[made], minlength=10)
print("new mistakes by the second-largest's position:", np.bincount(o[made, -2], minlength=10).tolist())

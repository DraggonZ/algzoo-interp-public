"""Question: zeroing the late gate's weights into the memory right after each overtaking costs 6 points
(tail_interv.log) but barely moves the old record's profile. What does it change? (1) what the mistakes name;
(2) the per-unit effect of the new record (nudge x_q alone, mean) at each position, with and without the late gate's
writes at q+1, q+2, for the placements of tail_route.py. 400,000 lists (seed 5) for (1)."""
import numpy as np
from memlib import lists
from tail_interv import run_iv
X = lists(5, 400_000); o = X.argsort(1); y = o[:, -2]
for name, w, off in [("as is", [], []), ("late gate 9 zeroed at q+1, q+2", [9], [1, 2])]:
    P = run_iv(X, w, off).argmax(1); bad = P != y
    print(f"{name:32s}: accuracy {100 * (1 - bad.mean()):.1f}; mistakes name the largest {100 * (P == o[:, -1])[bad].mean():.0f}%, the third {100 * (P == o[:, -3])[bad].mean():.0f}%, other {100 * ((P != o[:, -1]) & (P != o[:, -3]))[bad].mean():.0f}%")
Xa = np.random.default_rng(3).standard_normal((1_500_000, 10)); eps = 1e-4
for r, q in [(1, 4), (2, 5), (2, 7), (4, 7)]:
    pre = np.maximum(Xa[:, :r].max(1), 0) if r else np.zeros(len(Xa))
    ok = (Xa[:, r] > pre + 0.05) & (Xa[:, r] > 0.3) & (Xa[:, r + 1:q].max(1) < Xa[:, r]) & (Xa[:, q] > Xa[:, r] + 0.05)
    if q < 9: ok &= Xa[:, q + 1:].max(1) < Xa[:, q]
    Xs = Xa[ok][:100_000]; dX = np.zeros_like(Xs); dX[:, q] = eps
    print(f"== new record at {q} (old at {r}): per-unit effect of x_q, positions 0-9")
    for name, w, off in [("as is", [], []), ("late gate zeroed q+1, q+2", [9], [1, 2])]:
        d = (run_iv(Xs + dX, w, off) - run_iv(Xs, w, off)) / eps
        acc = (run_iv(Xs, w, off).argmax(1) == Xs.argsort(1)[:, -2]).mean()
        print(f"   {name:26s}", " ".join(f"{x:6.1f}" for x in d.mean(0)), f"| accuracy on these lists {100 * acc:.1f}")

"""Follow-up to hypothesis C ([0 x8, 4.0, 0.25] names a zero): prediction, the largest's imperfectly cancelled wave is
proportional to the largest, so when the second is tiny next to it (largest/second large) the leftover beats the
second's vote and accuracy falls. Accuracy on random lists by largest/second, and where the wrong answers sit relative
to the largest."""
import sys; sys.path.insert(0, "../steer")
from harness import *
X = lists(4_000_000, seed=17); o = X.argsort(1, descending=True); v = X.sort(1, descending=True).values
a = run(ARC, X).argmax(1); ok = a == o[:, 1]; q = v[:, 0] / v[:, 1]; pos2 = v[:, 1] > 0
for lo, hi in ((1, 1.1), (1.1, 1.5), (1.5, 2), (2, 3), (3, 5), (5, 10), (10, 1e9)):
    m = pos2 & (q >= lo) & (q < hi)
    if m.sum() < 100: continue
    wr = m & ~ok; off = (a[wr] - o[wr, 0])
    near_L = ((off.abs() <= 2) & (a[wr] != o[wr, 0])).double().mean().item()
    print(f"largest/second {lo:>4}-{hi:<6g} ({100*m.double().mean().item():5.2f}% of lists): right {100*ok[m].double().mean().item():5.1f}% | wrong answers within 2 of the largest (not it) {100*near_L:4.0f}%, the largest itself {100*(a[wr] == o[wr, 0]).double().mean().item():3.0f}%")

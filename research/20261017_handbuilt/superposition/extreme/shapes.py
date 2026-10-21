"""Shapes of confident mistakes (search3.log). Predictions, written first:
 S1 the largest arrives at the last step and must reveal the old record (the second) in one step; the older the second,
    the weaker the reveal, so on clear-gap lists the error rate rises with the second's age, worst at position 0.
 S2 the third right before the second (adjacent, earlier) beats it even with a 20% gap more often than other offsets."""
import sys; sys.path.insert(0, "../steer")
from harness import *
X = lists(16_000_000, seed=77); v = X.sort(1, descending=True).values
gap = (v[:, 1] >= 0.3 * v[:, 0]) & (v[:, 0] >= 1.2 * v[:, 1]) & (v[:, 1] >= 1.2 * v[:, 2]) & (v[:, 2] > 0.2)
X = X[gap]; o = X.argsort(1, descending=True); L, S, T = o[:, 0], o[:, 1], o[:, 2]; a = run(ARC, X).argmax(1); wr = a != S
print(f"{len(X):,} clear-gap lists, wrong {100*wr.double().mean().item():.2f}%")
print("S1: largest at step 9, second before the third (the second was the record), error by the second's position:")
for p in range(8):
    m = (L == 9) & (S == p) & (T > S)
    if m.sum() > 300: print(f"   second at {p}: wrong {100*wr[m].double().mean().item():5.2f}% (n {int(m.sum()):,}); third named {100*(a[m] == T[m]).double().mean().item():.2f}%")
m = (L < 9); print(f"   for comparison, largest not at step 9: wrong {100*wr[m].double().mean().item():.2f}%")
print("S2: error by the third's offset from the second (third position - second position):")
for d in (-3, -2, -1, 1, 2, 3):
    m = (T - S) == d
    print(f"   offset {d:+d}: wrong {100*wr[m].double().mean().item():5.2f}%, third named {100*(a[m] == T[m]).double().mean().item():.2f}% (n {int(m.sum()):,})")
print("S2 split, third right before the second (offset -1), by where the largest is:")
for lab, m in (("largest before both", (T - S == -1) & (L < T)), ("largest after both (both were records)", (T - S == -1) & (L > S))):
    print(f"   {lab:40s}: wrong {100*wr[m].double().mean().item():5.2f}% (n {int(m.sum()):,})")
print("S1 split, second at position 0, by the largest's position:")
for p in range(1, 10):
    m = (S == 0) & (L == p); print(f"   largest at {p}: wrong {100*wr[m].double().mean().item():5.2f}% (n {int(m.sum()):,})")

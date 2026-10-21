"""Follow-up to hypothesis D ([2.062, 0 x8, 2.0] names the largest): the account's smooth comparators start switching
the hiding off when a number reaches about 0.9 of the largest, so a second close to the largest partly reveals it.
Prediction: on two spikes with zeros elsewhere, the share of placements naming the largest rises as the ratio
largest/second falls toward 1, starting somewhere below 1.1."""
import sys; sys.path.insert(0, "../steer")
import itertools
from harness import *
P = [(i, j) for i, j in itertools.permutations(range(10), 2)]
for r in (1.5, 1.2, 1.1, 1.07, 1.05, 1.03, 1.02, 1.01, 1.001):
    X = torch.zeros(len(P), 10, device=DEV, dtype=dt)
    for k, (i, j) in enumerate(P): X[k, i] = 2.0 * r; X[k, j] = 2.0
    a = run(ARC, X).argmax(1).cpu().tolist()
    wrongL = [(i, j) for (i, j), x in zip(P, a) if x == i]; right = sum(x == j for (i, j), x in zip(P, a))
    print(f"largest/second {r:<6}: right at {right}/90, names the largest at {len(wrongL)}: {wrongL[:10]}")

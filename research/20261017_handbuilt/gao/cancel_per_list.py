"""Where are the mistakes? Per list, the correct answer's lead over the best other position, and the pushes on that
lead (sum over neurons of |(w_o[correct] - w_o[best other]) h_n|). A mistake is a negative lead. Bins of
|lead| / pushes: share of lists and share of all mistakes in each bin; and how well |lead| alone or the ratio
predicts a mistake within near-tie lists. ARC's network, 200,000 lists."""
import numpy as np
import simlib

X = simlib.lists(2121, 200_000); y = X.argsort(1)[:, -2]; r = np.arange(len(X))
L, _, H = simlib.run(X); h = H[:, 9]
other = L.copy(); other[r, y] = -np.inf; j = other.argmax(1)
lead = L[r, y] - L[r, j]; push = np.abs((simlib.WO[y] - simlib.WO[j]) * h).sum(1); ratio = np.abs(lead) / np.maximum(push, 1e-12)
wrong = lead < 0
print(f"accuracy {1 - wrong.mean():.4f}; median pushes {np.median(push):.0f}")
edges = [0, 0.005, 0.01, 0.02, 0.05, 0.1, 1.0]
for a, b in zip(edges[:-1], edges[1:]):
    m = (ratio >= a) & (ratio < b)
    print(f"|lead| / pushes {a:.3f}-{b:.3f}: {m.mean():.3f} of lists, {wrong[m].sum() / wrong.sum():.3f} of mistakes, mistake rate {wrong[m].mean():.3f}")
for a, b in ((0, 2), (2, 5), (5, 10), (10, 1e9)):
    m = (np.abs(lead) >= a) & (np.abs(lead) < b)
    print(f"|lead| {a}-{b}: {m.mean():.3f} of lists, {wrong[m].sum() / wrong.sum():.3f} of mistakes")

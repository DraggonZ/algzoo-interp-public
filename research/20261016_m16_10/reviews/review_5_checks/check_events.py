"""Review 5 check D (diagnostic, sampled): how many mechanism-level cases carry the mistakes, compared with linear regions?

Each input gets a case: the step t and kind (near M / near m) of its closest arrival comparison, gap bin, and the ages of
largest and second at the end. Counts cases needed to cover the mistakes, and how much P(wrong | gap bin) moves with t.
"""
import sys
from collections import Counter

import numpy as np

from algzoo.core import example, zoo

name = sys.argv[1] if len(sys.argv) > 1 else "example"
params = example() if name == "example" else zoo(16, 10, int(name))
w_hi, w_hh, w_oh = (np.asarray(p, np.float64) for p in params)
rng = np.random.default_rng(12)
N = 1 << 21
X = rng.standard_normal((N, 10))
hh = np.zeros((N, 16))
for t in range(10):
    hh = np.maximum(0, hh @ w_hh.T + X[:, t, None] * w_hi)
pred = (hh @ w_oh.T).argmax(1)
o = np.argsort(X, 1); p1, p2 = o[:, -1], o[:, -2]
wrong = pred != p2
gap = np.full(N, 9.0); gt = np.zeros(N, int); kind = np.zeros(N, int)
for t in range(1, 10):
    srt = np.sort(X[:, :t], 1)
    M = srt[:, -1]; m = srt[:, -2] if t >= 2 else np.full(N, -np.inf)
    x = X[:, t]
    for k, ref in [(0, M), (1, m)]:
        g = np.abs(x - ref)
        b = g < gap
        gap[b] = g[b]; gt[b] = t; kind[b] = k
gb = np.digitize(gap, [0.01, 0.03, 0.1, 0.3])
neg = X[np.arange(N), p2] < 0
print(f"model {name}: accuracy {1 - wrong.mean():.4f}; mistakes with min arrival gap < 0.1: {np.mean(gap[wrong] < 0.1):.3f}; negative second: {np.mean(neg[wrong]):.3f}")
print("P(wrong) by gap bin (rows) and step of the closest comparison t = 1..9 (columns), comparisons with the second only:")
for b, lab in enumerate(["<0.01", "0.01-0.03", "0.03-0.1", "0.1-0.3"]):
    row = [np.mean(wrong[(gb == b) & (gt == t) & (kind == 1) & ~neg]) if np.any((gb == b) & (gt == t) & (kind == 1) & ~neg) else np.nan for t in range(1, 10)]
    print(f"  {lab:>9}: " + " ".join(f"{v:5.3f}" for v in row))
print("same, comparisons with the largest:")
for b, lab in enumerate(["<0.01", "0.01-0.03", "0.03-0.1", "0.1-0.3"]):
    row = [np.mean(wrong[(gb == b) & (gt == t) & (kind == 0) & ~neg]) for t in range(1, 10)]
    print(f"  {lab:>9}: " + " ".join(f"{v:5.3f}" for v in row))
case = [(int(a), int(b), int(c), int(d), int(e)) for a, b, c, d, e in zip(gt[wrong], kind[wrong], gb[wrong], p1[wrong], p2[wrong])]
cnt = Counter(case)
c = np.sort(np.array(list(cnt.values())))[::-1]; cs = np.cumsum(c) / c.sum()
print(f"cases (t, kind, gap bin, position of largest, position of second): {len(cnt)} hold mistakes")
for f in [0.5, 0.75, 0.9]:
    print(f"  cases needed to cover {f:.0%} of mistakes: {np.searchsorted(cs, f) + 1}")

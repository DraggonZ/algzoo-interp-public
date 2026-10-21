"""The most confidently wrong lists so far share one shape: the positive numbers are tiny (about 0.01) and the negatives
large (about -2). Prediction: negative numbers drive the hiding neurons (input weights -10 to -13), so they add noise
that scales with their size while the votes scale with the positive values; accuracy should fall as
(size of the negatives) / (second's value) grows. Random lists with the second positive, binned by the mean absolute
value of the negative numbers divided by the second. Also: does the same hold for the smaller positive numbers?"""
import sys; sys.path.insert(0, "../steer")
from harness import *
X = lists(8_000_000, seed=5); o = X.argsort(1, descending=True); v = X.sort(1, descending=True).values; n = torch.arange(len(X), device=DEV)
a = run(ARC, X).argmax(1); ok = a == o[:, 1]; s = v[:, 1]
negsz = torch.where(X < 0, -X, torch.zeros_like(X)).sum(1) / (X < 0).sum(1).clamp(min=1); q = negsz / s
m0 = s > 0
for lo, hi in ((0, 0.5), (0.5, 1), (1, 2), (2, 5), (5, 10), (10, 30), (30, 100), (100, 1e9)):
    m = m0 & (q >= lo) & (q < hi)
    if m.sum() < 200: continue
    print(f"mean |negative| / second {lo:>4}-{hi:<6g} ({100*m.double().mean().item():5.2f}% of lists): right {100*ok[m].double().mean().item():5.1f}%", flush=True)

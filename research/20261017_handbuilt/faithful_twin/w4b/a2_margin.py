"""w4b analysis read: the hiding margin delivered through C1 (margin=True) against the detector's direct start
(w4b_cancel). 400,000 lists. Same function except where the design says (the number two steps after a record lies
within c1 d below the largest)? Prints accuracy, answers changed, and where the changed lists have their largest."""
import numpy as np
from algzoo.handbuilt_w4b import build_w4b

rng = np.random.default_rng(4); X = rng.standard_normal((400_000, 10))
def run(wi, wh, wo):
    h = np.zeros((len(X), len(wi)))
    for t in range(10): h = np.maximum(h @ wh.T + X[:, t:t + 1] * wi, 0)
    return h @ wo.T
truth = np.argsort(-X, 1)[:, 1]
A = run(*build_w4b(cancel=True)[:3]); import sys; sh = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0
B = run(*build_w4b(cancel=True, margin=True, share6=sh)[:3]); print('share6', sh)
a, b = A.argmax(1), B.argmax(1); ch = a != b
print(f"cancel: {100 * (a == truth).mean():.2f}%  margin via C1: {100 * (b == truth).mean():.2f}%  answers changed {100 * ch.mean():.3f}%")
r = X.argmax(1); M = X.max(1)
pre = np.maximum.accumulate(X, 1)
for pos in range(10):
    m = r == pos
    print(f"  largest at {pos}: lists {m.mean():.3f}, changed {100 * ch[m].mean():.3f}%, right {100 * (a[m] == truth[m]).mean():.2f} -> {100 * (b[m] == truth[m]).mean():.2f}")
# the designed exception: number two steps after the final largest within c1 d of it
d = np.array([M[i] - max(0, pre[i, r[i] - 1]) if r[i] > 0 else M[i] for i in range(len(X))])
x2 = np.array([X[i, r[i] + 2] if r[i] + 2 < 10 else -np.inf for i in range(len(X))])
exc = (x2 > M - 0.95 * d)
print(f"designed exception (step after next within 0.95 d below the largest): {100 * exc.mean():.2f}% of lists; changed there {100 * ch[exc].mean():.2f}%, elsewhere {100 * ch[~exc].mean():.3f}%")
print("score difference on unchanged-exception lists: max", np.abs(A - B)[~exc].max())

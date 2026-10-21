"""Question: when v18 at the trained gain scale (g = 10) errs, what does it name, and does the error follow a partial
switch of its hiding group? Compare with g = 10,000 (no partial switching). Also, for comparison, the trained network."""
import numpy as np
from algzoo.core import example
from algzoo.handbuilt_v18 import build_v18
rng = np.random.default_rng(8); X = rng.standard_normal((300000, 10)); N = len(X); y = np.argsort(X, 1)[:, -2]
order = np.argsort(-X, 1); rank = np.argsort(order, 1)        # rank[i, p] = 0 for the largest
def run(P, hid):
    P = [np.asarray(p, float) for p in P]; h = np.zeros((N, len(P[0]))); part = np.zeros(N, bool)
    for t in range(10):
        h = np.maximum(h @ P[1].T + X[:, t:t + 1] * P[0], 0); o = (h[:, hid] == 0).sum(1); part |= (o > 0) & (o < len(hid)) & (X[:, t] > 0)
    return (h @ P[2].T).argmax(1), part
net = build_v18(); I = net.idx; hid = [I[k] for k in ("clock0", "clock1", "clock2", "flip")]
for name, P, hd in [("v18 g 10,000", build_v18().params, hid), ("v18 g 10", build_v18(g=10, a=10, w=10, lam=10.0, c=-9.0, eps=0.1).params, hid),
                    ("trained", example(), [1, 6, 7, 8])]:
    a, part = run(P, hd); wrong = a != y; rk = rank[np.arange(N), a]
    print(f"{name:13s} accuracy {1 - wrong.mean():.4f}; errors naming the largest {(wrong & (rk == 0)).mean() / wrong.mean():.2f}, "
          f"the third {(wrong & (rk == 2)).mean() / wrong.mean():.2f}, lower {(wrong & (rk > 2)).mean() / wrong.mean():.2f}; "
          f"lists with a partial switch {part.mean():.2f}: accuracy there {1 - wrong[part].mean():.3f}, elsewhere {1 - wrong[~part].mean():.3f}")

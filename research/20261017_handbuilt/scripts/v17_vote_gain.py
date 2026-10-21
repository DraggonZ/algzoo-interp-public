"""Own-score response d score_q / d x_q for a kept non-record y at q = r + k after a record P at r, split by
whether P is later overtaken (y < P either way). Median over random lists; finite difference."""
import numpy as np, torch as th, sys
sys.path.insert(0, "scripts")
from algzoo.core import example, forward
from handbuilt_v16 import build_v16, scores
X = np.random.default_rng(33).standard_normal((400_000, 10)); B = len(X)
m1p = np.stack([np.max(np.c_[np.zeros((B, 2)), X[:, :t]], 1) for t in range(10)], 1); rec = X > m1p
nets = {"trained": lambda Z: forward(example(), th.tensor(Z)).numpy(), "v16 x100": lambda Z: 100 * scores(build_v16(gamma=0.25), Z)}
eps = 0.01
for k in (1, 2, 3):
    for q in (4, 5):
        r = q - k
        base = rec[:, r] & ~rec[:, r + 1:q + 1].any(1) & (X[:, q] > 0.5 * X[:, r]) & (X[:, q] < X[:, r] - 2 * eps)
        over = base & rec[:, q + 1:].any(1); stay = base & ~rec[:, q + 1:].any(1)
        line = f"y at r+{k} (q={q}):"
        for name, f in nets.items():
            out = []
            for mm in (stay, over):
                Z = X[mm]; Z2 = Z.copy(); Z2[:, q] += eps
                out.append(np.median((f(Z2)[:, q] - f(Z)[:, q]) / eps))
            line += f"  {name}: record stays {out[0]:5.1f}, overtaken later {out[1]:5.1f}"
        print(line)

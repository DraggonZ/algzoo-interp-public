"""Fix the store's account numbers (40% low with one canonical list per arrangement). Prediction: the store's effect
depends on the largest's size relative to the contenders, so canonical lists per arrangement and per cell of
second/largest (4 or 8 equal-probability cells) should bring the predicted answer changes to within 30%."""
import sys; sys.path.insert(0, "../level2/deduce")
from harness import *
from scipy.stats import norm
from case_maps import A, idx
from case_maps2 import cells, qL
X = lists(1_000_000, seed=41); o = X.argsort(1, descending=True); v = X.sort(1, descending=True).values
a_idx = torch.tensor([idx[q] for q in zip(o[:, 0].tolist(), o[:, 1].tolist(), o[:, 2].tolist())], device=DEV)
S0 = run(ARC, X); a0 = S0.argmax(1)
for name, nd in {"store x1.01 at step 5": [([2, 4], {5}, 1.01)], "balanced memory x1.02, store x1.019": [(GROUPS["memory"], {5}, 1.02), ([2, 4], {5}, 1.0192)]}.items():
    real = run(ARC, X, nd).argmax(1); out = [f"real {100*(real != a0).double().mean().item():.2f}%"]
    for k in (1, 4, 8):
        e1, e2, m1, m2 = cells(k); C = []
        for (l, s, t) in A:
            for i in range(k):
                xs = qL * m1[i].item(); xt = 0.656 / 1.001 * xs; rest = -norm.pdf(xt) / norm.cdf(xt)
                c = [rest] * 10; c[l], c[s], c[t] = qL, xs, xt; C.append(c)
        C = torch.tensor(C, device=DEV, dtype=dt); dC = first_order(ARC, C, nd) - run(ARC, C)
        i = torch.bucketize(v[:, 1] / v[:, 0], e1[1:-1].contiguous()); cell = a_idx * k + i
        acc = (S0 + (v[:, 0] / qL)[:, None] * dC[cell]).argmax(1)
        out.append(f"{k} cells: {100*(acc != a0).double().mean().item():.2f}%")
    print(f"{name:38s} answers changed: " + " | ".join(out), flush=True)

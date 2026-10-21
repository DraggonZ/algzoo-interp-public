"""w7b analysis read 2: where w6b's extra errors come from (largest at positions 0, 1, 9) and what w7b names there.
For lists w6b gets wrong and w7b right: the named position's offset from the largest, and whether the largest was
overtaken (it never is: the largest of the list stands from its arrival) -- so the spill is from earlier records."""
import numpy as np
from collections import Counter
from algzoo.handbuilt_w6b import build_w6b
from algzoo.handbuilt_w7b import build_w7b


def ans(W, x):
    wi, wh, wo = W[:3]; h = np.zeros((len(x), len(wi)))
    for t in range(10): h = np.maximum(h @ wh.T + x[:, t:t + 1] * wi, 0)
    return (h @ wo.T).argmax(1)


x = np.random.default_rng(3).standard_normal((400_000, 10))
truth = np.argsort(-x, 1)[:, 1]; p1 = x.argmax(1)
a6, a7 = ans(build_w6b(), x), ans(build_w7b(), x)
fix = (a6 != truth) & (a7 == truth); brk = (a6 == truth) & (a7 != truth)
print(f"fixed by w7b {100 * fix.mean():.2f}% of lists, broken {100 * brk.mean():.2f}%")
for name, m, a_ in (("fixed: w6b named", fix, a6), ("broken: w7b named", brk, a7)):
    print(name, "offset from the largest (named - largest):", Counter((a_[m] - p1[m]).tolist()).most_common(8))
    print(name, "offset from the true second (named - second):", Counter((a_[m] - truth[m]).tolist()).most_common(8))
    print(name, "named = largest:", f"{100 * (a_[m] == p1[m]).mean():.1f}%")

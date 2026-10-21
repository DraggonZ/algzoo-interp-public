"""w12b analysis 2: where keep-first gains over w8b. Lists that only one of the two networks gets right, by (largest's
position, second's position), and what w8b named on the lists only keep-first gets right."""
import numpy as np
from algzoo.handbuilt_w8b import build_w8b
from algzoo.handbuilt_w12b import free_linear
from collections import Counter


def run(wi, wh, wo, x):
    h = np.zeros((len(x), len(wi)))
    for t in range(10):
        h = np.maximum(h @ wh.T + x[:, t:t + 1] * wi, 0)
    return np.argmax(h @ wo.T, 1)


x = np.random.default_rng(1).standard_normal((1_000_000, 10))
o = np.argsort(x, 1); sec, top = o[:, -2], o[:, -1]
wi, wh, wo, idx, _ = build_w8b()
p0 = run(wi, wh, wo, x)
p1 = run(*free_linear(wi, wh, wo, idx)[:3], x)
g = (p1 == sec) & (p0 != sec); l = (p0 == sec) & (p1 != sec)
print(f"only keep-first right: {g.sum()}  only w8b right: {l.sum()}  (of {len(x)})")
print("gains by (largest pos, second pos):", Counter(zip(top[g], sec[g])).most_common(8))
print("losses by (largest pos, second pos):", Counter(zip(top[l], sec[l])).most_common(8))
print("on gains, w8b named:", Counter(p0[g]).most_common(5))
print("on losses, keep-first named:", Counter(p1[l]).most_common(5))
# is the named number below the keep cut (0.75 x the runner-up so far when it arrived)?
def below_cut(pos):
    r = np.empty(len(pos), bool)
    for i, (row, j) in enumerate(zip(x, pos)):
        prev = np.sort(row[:j])
        r[i] = j >= 2 and row[j] < 0.75 * prev[-2]
    return r
gi = np.flatnonzero(g)[:20000]
print("on gains, w8b's named number was below the keep cut when it arrived:", f"{100 * below_cut(p0[gi]) .mean():.1f}%",
      "(on", len(gi), "lists)")
li = np.flatnonzero(l)[:20000]
print("on losses, the true second was below the keep cut when it arrived:", f"{100 * below_cut(sec[li]).mean():.1f}%")

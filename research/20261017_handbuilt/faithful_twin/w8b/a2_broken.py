"""w8b analysis read: the lists w8b breaks against w7b. What does w8b name (largest, a neighbour of the second), and
is there a record followed at once by a new second-largest (the case where the start is now the gap, not the jump)?"""
import numpy as np
from algzoo.handbuilt_w7b import build_w7b
from algzoo.handbuilt_w8b import build_w8b

A = build_w7b(); B = build_w8b()
rng = np.random.default_rng(3); n = 400_000; x = rng.standard_normal((n, 10))
def run(W):
    wi, wh, wo = W[:3]; h = np.zeros((n, len(wi)))
    for t in range(10): h = np.maximum(h @ wh.T + x[:, t:t + 1] * wi, 0)
    return (h @ wo.T).argmax(1)
ya, yb = run(A), run(B); o = np.argsort(-x, 1); top, sec = o[:, 0], o[:, 1]
br = (ya == sec) & (yb != sec); fx = (ya != sec) & (yb == sec)
print(f"broken {br.sum()}  fixed {fx.sum()}")
print(f"  broken: names the largest {100 * (yb[br] == top[br]).mean():.0f}%, offset named - second: " +
      " ".join(f"{k}:{v}" for k, v in zip(*np.unique(yb[br] - sec[br], return_counts=True)) if v > 0.03 * br.sum()))
# a record at t followed by x_{t+1} between the previous largest and the record ('close follower'), for the final largest
m = np.maximum.accumulate(x, 1)
def close_follower(i):
    t = top[i]
    if t == 0 or t == 9: return False
    return m[i, t - 1] < x[i, t + 1] < x[i, t]
cf = np.array([close_follower(i) for i in np.flatnonzero(br)])
allcf = np.array([close_follower(i) for i in range(20000)])
print(f"  broken lists where the final largest is followed at once by the new second: {100 * cf.mean():.0f}% (all lists: {100 * allcf.mean():.0f}%)")
print(f"  broken by position of the largest: " + " ".join(f"{k}:{v}" for k, v in zip(*np.unique(top[br], return_counts=True))))
print(f"  broken by second - largest position: " + " ".join(f"{k}:{v}" for k, v in zip(*np.unique(sec[br] - top[br], return_counts=True)) if v > 0.03 * br.sum()))

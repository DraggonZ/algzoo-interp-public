"""Sort ARC's mistakes by kind and gap. Kinds: at most one positive number; answer = the
largest; answer = the third; answer = fourth or lower. Gap = (second - answered number) / |second|."""
import numpy as np, torch as th
from algzoo.core import example, forward

X = np.random.default_rng(25).standard_normal((2_000_000, 10))
P = forward(example(), th.tensor(X)).numpy().argmax(1)
o = X.argsort(1)[:, ::-1]; r = np.arange(len(X))
rank = np.argmax(o == P[:, None], 1)            # 0 largest, 1 second, ...
wrong = rank != 1
onepos = (X > 0).sum(1) <= 1
sec = X[r, o[:, 1]]; ans = X[r, P]
gap = np.abs(sec - ans) / np.abs(sec).clip(1e-9)
print(f"error rate {wrong.mean():.4f}; shares of all mistakes:")
kinds = {"at most one positive number": wrong & onepos,
         "names the largest": wrong & ~onepos & (rank == 0),
         "names the third": wrong & ~onepos & (rank == 2),
         "names the fourth or lower": wrong & ~onepos & (rank >= 3)}
for k, m in kinds.items():
    g = gap[m]
    print(f"  {k:28s} {m.sum() / wrong.sum():.3f}   gap under 3%: {(g < .03).mean():.2f}, 3-10%: {((g >= .03) & (g < .1)).mean():.2f}, 10-40%: {((g >= .1) & (g < .4)).mean():.2f}, over 40%: {(g >= .4).mean():.2f}")
m = wrong & ~onepos & (rank == 0) & (gap >= .4)
print(f"  of 'names the largest' with gap over 40%: second-largest negative or near zero (< 0.2) in {(sec[m] < 0.2).mean():.2f}")

"""Do the soft-dip v3 networks fail like ARC? Error rate by the second-third gap (relative), by whether the second
and third sit next to each other, agreement with ARC and share of ARC's mistakes repeated."""
import numpy as np, torch as th
from algzoo.core import example, forward
from handbuilt_v3 import build_v3, scores

rng = np.random.default_rng(11)
X = rng.standard_normal((1_000_000, 10))
o = X.argsort(1); tgt, third = o[:, -2], o[:, -3]
srt = np.sort(X, 1); gap = (srt[:, -2] - srt[:, -3]) / np.abs(srt[:, -2]).clip(1e-9)
adj = np.abs(tgt - third) == 1
arc = forward(example(), th.tensor(X)).numpy().argmax(1)
bins = [0, .01, .03, .1, .2, .4, 10]
nets = {"ARC": arc}
for w in (0.8, 0.77, 0.75, 0.72, 0.7):
    nets[f"v3 dip {w}"] = scores(build_v3(w=w), X).argmax(1)
print("error rate by second-third gap (share of the second's value); last column: share of all mistakes that are second-third")
print(f"{'':12s} {'acc':>6s} " + " ".join(f"{bins[i]:.2f}-{bins[i+1]:.2f}" for i in range(len(bins) - 1)) + "  adjacent  apart   agree  ARC-mistakes")
w_arc = arc != tgt
for name, P in nets.items():
    row = [1 - (P[(gap >= bins[i]) & (gap < bins[i + 1])] == tgt[(gap >= bins[i]) & (gap < bins[i + 1])]).mean() for i in range(len(bins) - 1)]
    near = gap < .1
    print(f"{name:12s} {(P == tgt).mean():.4f} " + " ".join(f"{r:9.3f}" for r in row) +
          f"  {1 - (P[adj & near] == tgt[adj & near]).mean():8.3f} {1 - (P[~adj & near] == tgt[~adj & near]).mean():6.3f}"
          f"  {(P == arc).mean():.3f}  {(P[w_arc] == arc[w_arc]).mean():.2f}")

"""Small vote-size errors in v3: multiply each position's readout (all its output weights) by 1 + sigma * N(0, 1),
fixed per network (5 networks per sigma). Error by second-third gap, neighbours versus apart, against ARC."""
import numpy as np, torch as th
from algzoo.core import example, forward
from algzoo.handbuilt import build_v3, run

rng = np.random.default_rng(13)
X = rng.standard_normal((600_000, 10))
o = X.argsort(1); tgt, third = o[:, -2], o[:, -3]
srt = np.sort(X, 1); gap = (srt[:, -2] - srt[:, -3]) / np.abs(srt[:, -2]).clip(1e-9)
adj = np.abs(tgt - third) == 1
arc = forward(example(), th.tensor(X)).numpy().argmax(1)
net = build_v3()
H = run(net.params, X)[:, -1]
bins = [0, .01, .03, .1, .2, 10]


def row(P):
    r = [1 - (P[(gap >= bins[i]) & (gap < bins[i + 1])] == tgt[(gap >= bins[i]) & (gap < bins[i + 1])]).mean() for i in range(len(bins) - 1)]
    near = gap < .1
    return (P == tgt).mean(), r, 1 - (P[adj & near] == tgt[adj & near]).mean(), 1 - (P[~adj & near] == tgt[~adj & near]).mean()


hdr = " ".join(f"{bins[i]:.2f}-{bins[i+1]:.2f}" for i in range(len(bins) - 1))
print(f"{'':16s} {'acc':>6s} {hdr}  adjacent apart")
a, r, ad, ap = row(arc)
print(f"{'ARC':16s} {a:.4f} " + " ".join(f"{v:9.3f}" for v in r) + f"  {ad:8.3f} {ap:5.3f}")
for sigma in (0.0, 0.01, 0.02, 0.03, 0.05):
    res = []
    for seed in range(5):
        g = 1 + sigma * np.random.default_rng(100 + seed).standard_normal(10)
        res.append(row((H @ (net.w_oh * g[:, None]).T).argmax(1)))
    a = np.mean([x[0] for x in res]); r = np.mean([x[1] for x in res], 0)
    ad = np.mean([x[2] for x in res]); ap = np.mean([x[3] for x in res])
    print(f"{'v3 gains ±' + str(int(sigma * 100)) + '%':16s} {a:.4f} " + " ".join(f"{v:9.3f}" for v in r) + f"  {ad:8.3f} {ap:5.3f}")

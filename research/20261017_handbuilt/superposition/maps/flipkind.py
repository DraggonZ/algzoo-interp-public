"""Test 5: at each one-change flip, does any neuron change on/off state (any step), or does the margin cross zero linearly?
If linear, report the swept number's slope on the second-vs-third lead there (lead change per unit of the number)."""
import numpy as np, torch as th
from collections import Counter
from algzoo.core import example, forward
P = [np.asarray(p, float) for p in example()]
rng = np.random.default_rng(11); L = []
while len(L) < 300:
    x = rng.standard_normal(10); v = np.sort(x)[::-1]
    if v[1] > 0 and v[1] - v[2] < 0.05 * v[1]: L.append(x)
S = 401; same = 0; diffs = Counter(); n = 0; slopes = []; leads_lo = []
for x in L:
    o = np.argsort(-x); third = x[o[2]]; g = np.linspace(-3, 0.9 * third, S)
    for p in o[3:]:
        Y = np.repeat(x[None], S, 0); Y[:, p] = g; Yt = th.tensor(Y)
        lg, hs = forward(P, Yt, return_hidden=True); a = lg.argmax(-1).numpy(); ch = np.nonzero(np.diff(a))[0]
        if len(ch) != 1: continue
        k = ch[0]; n += 1
        on = (hs[[k, k + 1], 1:] > 0).numpy()               # 2 x 10 x 16
        d = on[0] != on[1]
        if not d.any():
            same += 1
            lead = (lg[:, o[1]] - lg[:, o[2]]).numpy(); slopes.append((lead[k + 1] - lead[k]) / (g[k + 1] - g[k]))
        else:
            for t, j in zip(*np.nonzero(d)): diffs[(int(j), int(t) - int(p))] += 1
print(f"one-change flips: {n}; no neuron changes state at the flip: {same} ({100*same/n:.0f}%)")
if slopes:
    s = np.abs(np.array(slopes)); print(f"  linear flips: |slope of the lead per unit of the swept number| median {np.median(s):.2f}, quartiles {np.percentile(s,25):.2f}-{np.percentile(s,75):.2f}")
print("  where state changes happen (neuron, steps after the swept number), top 10:", diffs.most_common(10))

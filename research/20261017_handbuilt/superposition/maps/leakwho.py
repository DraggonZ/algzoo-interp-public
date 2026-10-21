"""Look 6b: which smaller numbers leak into ARC's second-vs-third lead? |slope| of the lead on each smaller number (exact
gradient), split by: kept or forgotten by the keep gate (neuron 5 on one step after it, in the actual run; position 9
has no later step), sign of the number, and whether it came before or after the largest. Same 400 lists as leakmap.py."""
import numpy as np, torch as th
from algzoo.core import example
P = [th.tensor(np.asarray(p, float)) for p in example()]
rng = np.random.default_rng(11); L = []
while len(L) < 400:
    x = rng.standard_normal(10); v = np.sort(x)[::-1]
    if v[1] > 0 and v[1] - v[2] < 0.05 * v[1]: L.append(x)
X = np.array(L); O = np.argsort(-X, 1); wi, wh, wo = P
x = th.tensor(X, requires_grad=True); h = th.zeros(len(X), 16, dtype=th.float64); H = []
for t in range(10): h = th.relu(h @ wh.T + x[:, t:t + 1] * wi); H.append(h)
s = h @ wo.T; n = th.arange(len(X)); lead = s[n, th.tensor(O[:, 1])] - s[n, th.tensor(O[:, 2])]
g, = th.autograd.grad(lead.sum(), x); g = np.abs(g.numpy()); H = th.stack(H, 1).detach().numpy()
rows = []
for i in range(len(X)):
    for p in O[i, 3:]:
        kept = None if p == 9 else H[i, p + 1, 5] > 0
        rows.append((g[i, p], kept, X[i, p] > 0, p > O[i, 0]))
R = np.array([(r[0], -1 if r[1] is None else int(r[1]), int(r[2]), int(r[3])) for r in rows])
def show(lab, m):
    v = R[m, 0]; print(f"  {lab:44s} n {m.sum():5d}  |slope| median {np.median(v):.2f}  mean {v.mean():.2f}  share of all |slope| {100*v.sum()/R[:,0].sum():4.1f}%")
print("ARC, |slope| of the second-vs-third lead on each smaller number (400 near ties):")
show("kept by the keep gate", R[:, 1] == 1); show("forgotten by the keep gate", R[:, 1] == 0); show("at position 9 (no later step)", R[:, 1] == -1)
show("positive number", R[:, 2] == 1); show("negative number", R[:, 2] == 0)
show("before the largest", R[:, 3] == 0); show("after the largest", R[:, 3] == 1)
for k in (1, 0):
    for b in (0, 1): show(f"{'kept' if k else 'forgotten'}, {'after' if b else 'before'} the largest", (R[:, 1] == k) & (R[:, 3] == b))

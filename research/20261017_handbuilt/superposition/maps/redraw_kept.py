"""Test 8, see README.md."""
import numpy as np, torch as th
from algzoo.core import example, forward
Pn = [np.asarray(p, float) for p in example()]
rng = np.random.default_rng(12); L = []
while len(L) < 4000:
    x = rng.standard_normal(10); v = np.sort(x)[::-1]
    if v[1] > 0 and v[1] - v[2] < 0.05 * v[1]: L.append(x)
X = np.array(L); O = np.argsort(-X, 1); lg, H = forward(Pn, th.tensor(X), return_hidden=True); H = H[:, 1:].numpy(); d0 = lg.argmax(-1).numpy()
r = np.random.default_rng(5)
def trunc(below):
    z = r.standard_normal()
    while z >= below: z = r.standard_normal()
    return z
A, B, keepn = [], [], []
for i, x in enumerate(X):
    small = list(O[i, 3:]); kept = [p for p in small if p < 9 and H[i, p + 1, 5] > 0]; forg = [p for p in small if p not in kept and p < 9]
    k = min(len(kept), len(forg)); keepn.append(k)
    ya, yb = x.copy(), x.copy()
    for p in (r.choice(kept, k, replace=False) if k else []): ya[p] = trunc(x[O[i, 2]])
    for p in (r.choice(forg, k, replace=False) if k else []): yb[p] = trunc(x[O[i, 2]])
    A.append(ya); B.append(yb)
keepn = np.array(keepn); use = (keepn > 0) & ((d0 == O[:, 1]) | (d0 == O[:, 2]))
for lab, Y in (("kept numbers redrawn", A), ("same number of forgotten numbers redrawn", B)):
    d = forward(Pn, th.tensor(np.array(Y))).argmax(-1).numpy()
    flip = ((d == O[:, 1]) != (d0 == O[:, 1]))[use]
    print(f"{lab:44s}: second-vs-third decisions that flip {100*flip.mean():.1f}% (lists with at least one kept smaller number: {use.sum()}, median count {np.median(keepn[use]):.0f})")

# look 8b: among flips after redrawing forgotten numbers, how often did a redrawn number become kept?
YB = np.array(B); lgb, HB = forward(Pn, th.tensor(YB), return_hidden=True); HB = HB[:, 1:].numpy(); db = lgb.argmax(-1).numpy()
became = np.array([any(YB[i, p] != X[i, p] and p < 9 and HB[i, p + 1, 5] > 0 for p in range(10)) for i in range(len(X))])
flipb = ((db == O[:, 1]) != (d0 == O[:, 1])) & use
print(f"forgotten-redraw lists where a redrawn number became kept: {100*became[use].mean():.1f}%; flip rate with one {100*flipb[became & use].sum()/max((became & use).sum(),1):.1f}%, without {100*flipb[~became & use].sum()/max((~became & use).sum(),1):.1f}%")

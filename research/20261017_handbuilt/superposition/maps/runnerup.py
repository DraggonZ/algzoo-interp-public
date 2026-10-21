"""Test 3, see README.md."""
import numpy as np, torch as th
from algzoo.core import example, forward
from algzoo.handbuilt_v23 import build_v23
NETS = {"ARC": [np.asarray(p, float) for p in example()],
        "hand-built": [np.asarray(p, float) for p in build_v23(U_mem=1.0, share=0.25, rho=0.99, level_exact=False).params]}
rng = np.random.default_rng(11); L = []
while len(L) < 300:
    x = rng.standard_normal(10); v = np.sort(x)[::-1]
    if v[1] > 0 and v[1] - v[2] < 0.05 * v[1]: L.append(x)
def runner_up(x, t):          # index of the second-largest among x[0..t-1], or -1
    return int(np.argsort(-x[:t])[1]) if t >= 2 else -1
S = 401; X, lab = [], []
for x in L:
    o = np.argsort(-x); third = x[o[2]]; g = np.linspace(-3, 0.9 * third, S)
    early, late = min(o[1], o[2]), max(o[1], o[2]); ru_e, ru_l = runner_up(x, early), runner_up(x, late)
    for p in o[3:]:
        Y = np.repeat(x[None], S, 0); Y[:, p] = g; X.append(Y)
        lab.append(("runner-up when the earlier contender arrived" if p == ru_e else
                    "runner-up when the later contender arrived" if p == ru_l else "other smaller number", o[1], o[2]))
X = th.tensor(np.concatenate(X))
for n, P in NETS.items():
    a = forward(P, X).argmax(-1).numpy().reshape(len(lab), S); ch = (np.diff(a, axis=1) != 0).sum(1)
    print(f"\n{n}:")
    for k in ("runner-up when the earlier contender arrived", "runner-up when the later contender arrived", "other smaller number"):
        idx = [i for i, l in enumerate(lab) if l[0] == k]; c = ch[idx]
        tt = ts = 0
        for i in idx:
            if ch[i] == 1:
                lo, hi, s2, s3 = a[i, 0], a[i, -1], lab[i][1], lab[i][2]
                tt += (lo == s2 and hi == s3); ts += (lo == s3 and hi == s2)
        print(f"  {k:46s} {len(idx):5d} sweeps: change at least once {100*np.mean(c > 0):5.1f}%; one-change sweeps raising it moves toward the third {tt}, toward the second {ts}")

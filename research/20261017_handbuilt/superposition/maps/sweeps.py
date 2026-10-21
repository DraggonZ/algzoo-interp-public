"""One threshold per list? See README.md, test 2."""
import numpy as np, torch as th
from collections import Counter
from algzoo.core import example, forward
from algzoo.handbuilt_v23 import build_v23
NETS = {"ARC": [np.asarray(p, float) for p in example()],
        "hand-built": [np.asarray(p, float) for p in build_v23(U_mem=1.0, share=0.25, rho=0.99, level_exact=False).params]}
rng = np.random.default_rng(11); L = []
while len(L) < 300:
    x = rng.standard_normal(10); v = np.sort(x)[::-1]
    if v[1] > 0 and v[1] - v[2] < 0.05 * v[1]: L.append(x)
S = 401; X, meta = [], []
for li, x in enumerate(L):
    o = np.argsort(-x); third = x[o[2]]; g = np.linspace(-3, third - 1e-6, S)
    for p in o[3:]:
        Y = np.repeat(x[None], S, 0); Y[:, p] = g; X.append(Y)
        early, late = min(o[1], o[2]), max(o[1], o[2])
        where = ("before both contenders" if p < early else "between the contenders" if p < late else "after both contenders")
        where += ", before the largest" if p < o[0] else ", after the largest"
        meta.append((li, p, o[1], o[2], where, x[p]))
X = th.tensor(np.concatenate(X))
for n, P in NETS.items():
    a = forward(P, X).argmax(-1).numpy().reshape(len(meta), S)
    ch = (np.diff(a, axis=1) != 0).sum(1); c = Counter(np.minimum(ch, 3))
    print(f"\n{n}: {len(meta)} sweeps; changes 0: {100*c[0]/len(meta):.1f}%  1: {100*c[1]/len(meta):.1f}%  2: {100*c[2]/len(meta):.1f}%  3+: {100*c[3]/len(meta):.1f}%;"
          f" lists with at least one changing sweep {100*np.mean([any(ch[i] for i in range(len(meta)) if meta[i][0] == li) for li in range(len(L))]):.0f}%")
    if n == "ARC":
        D = Counter(); T = Counter()
        for i, (li, p, s2, s3, where, xv) in enumerate(meta):
            if ch[i] != 1: continue
            lo, hi = a[i, 0], a[i, -1]
            d = "toward the third" if (lo == s2 and hi == s3) else "toward the second" if (lo == s3 and hi == s2) else "to or from another position"
            D[(where, d)] += 1; T[where] += 1
        tot = Counter(m[4] for m in meta)
        print("one-change sweeps, by where the swept number sits (count, share of all sweeps there), and direction when the number is raised:")
        for w in sorted(tot):
            parts = ", ".join(f"{d} {D[(w, d)]}" for d in ("toward the third", "toward the second", "to or from another position"))
            print(f"  {w:45s} {T[w]:4d} of {tot[w]:4d} ({100*T[w]/tot[w]:4.1f}%): {parts}")
        k = np.argsort(-ch)[:3]
        for i in k: print("  most changes:", ch[i], "list", meta[i][0], "position", meta[i][1], meta[i][4], "answers along sweep:", "".join(str(z) for z in a[i][::20]))

"""Case study: a degraded copy (mult0.02_8). Which pairs fail, where along the slide the vote is lost, and which neurons
switch differently from ARC's model at that step."""
import numpy as np

from algzoo.bench import load_bench
from algzoo.core import example

names, acc, models = load_bench(("mult",))
cp = tuple(np.asarray(q, np.float64) for q in models[list(names).index("mult0.02_8")])
base = tuple(np.asarray(q, np.float64) for q in example())
r = 0.7


def trace(params, x):
    w_hi, w_hh, w_oh = params
    h = np.zeros(16); S, P = [], []
    for t in range(10):
        pre = w_hh @ h + x[t] * w_hi; h = np.maximum(0, pre)
        s = w_oh @ h; S.append(s - s.mean()); P.append(pre > 0)
    return np.array(S), np.array(P)


fails = []
for p1 in range(10):
    for p2 in range(10):
        if p1 == p2:
            continue
        x = np.zeros(10); x[p1] = 1.0; x[p2] = r
        S, _ = trace(cp, x)
        if S[9].argmax() != p2:
            t0 = max(p1, p2)
            lost = next((t for t in range(t0, 10) if S[t].argmax() != p2 + 9 - t), None)
            fails.append((p1, p2, lost, S[9].argmax()))
print(f"mult0.02_8 at ratio {r}: {len(fails)} failing pairs (largest, second, step where vote lost, final answer):")
print("  ", fails)
from collections import Counter
print("vote lost at step:", Counter(f[2] for f in fails))
print("lost at age of largest:", Counter(f[2] - f[0] for f in fails), " age of second:", Counter(f[2] - f[1] for f in fails))
p1, p2, t_lost, ans = fails[0]
x = np.zeros(10); x[p1] = 1.0; x[p2] = r
Sc, Pc = trace(cp, x); Sb, Pb = trace(base, x)
np.set_printoptions(linewidth=200, precision=1, suppress=True)
print(f"\ncase largest {p1}, second {p2}: scores by step (copy | ARC's model)")
for t in range(max(p1, p2), 10):
    print(f"  t{t}: copy {Sc[t]}  | base {Sb[t]}")
    print(f"       neurons on in copy but not base: {list(np.where(Pc[t] & ~Pb[t])[0])}, on in base but not copy: {list(np.where(Pb[t] & ~Pc[t])[0])}")

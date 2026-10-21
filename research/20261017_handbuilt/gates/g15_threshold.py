"""The keep gate's threshold, measured in the network and predicted by the rule, on constructed lists.
List: every number -0.3 except the largest m1 at position i1, the second-largest m2 at position i2, and the tested
number x at position t (x_{t+1} = -0.3). Threshold = smallest x for which neuron 5 is on one step after x (bisection).
Network: the trained network. Rule: gaterules3 rule with switch points by 'roles' and D5 by 'gates' (full) and by
'ages' (m1 and m2 by age only). Also the threshold's response to nudging m2 and m1 by 0.05 (finite difference)."""
import pickle
import numpy as np
from gatelib import run
from gaterules3 import rule_values, HJ

co = pickle.load(open("g11_tables.pkl", "rb"))
SPECS = {"full": {**{f"r{j}": "roles" for j in HJ}, "D5": "gates", "D9": "gates"},
         "ages": {**{f"r{j}": "ages" for j in HJ}, "D5": "ages", "D9": "ages"}}


def build(m1, i1, m2, i2, t, xs):
    X = np.full((len(xs), 10), -0.3); X[:, i1] = m1; X[:, i2] = m2; X[:, t] = xs; return X


def thresholds(m1, i1, m2, i2, t):
    lo, hi = np.zeros(4), np.full(4, m1 * 0.999)   # network, full rule, ages rule
    out = []
    for who in ("net", "full", "ages"):
        a, b = 0.0, min(m1, 3.0) * 0.999
        for _ in range(30):
            mid = (a + b) / 2; X = build(m1, i1, m2, i2, t, np.array([mid]))
            if who == "net":
                on = run(X)[2][0, t + 1, 5] > 0
            else:
                on = rule_values(X, SPECS[who], co)[0][5][0, t + 1] > 0
            a, b = (a, mid) if on else (mid, b)
        out.append(b)
    return out


print("threshold for x (and threshold / m2); m1 = 1.5")
print(f"{'i1 i2 t':10s} {'m2':>4s}   {'network':>14s} {'rule full':>14s} {'rule ages':>14s}")
cases = [(0, 1, 3), (0, 1, 6), (0, 1, 8), (0, 4, 6), (0, 4, 8), (0, 6, 8), (2, 3, 5), (2, 3, 8), (4, 5, 8), (1, 0, 3), (1, 0, 7),
         (3, 1, 6), (5, 2, 8), (6, 3, 8)]
rows = []
for (i1, i2, t) in cases:
    for m2 in (0.6, 1.0):
        th = thresholds(1.5, i1, m2, i2, t); rows.append(th)
        print(f"{i1} {i2} {t}      {m2:4.1f}   " + " ".join(f"{v:6.3f} ({v / m2:4.2f})" for v in th))
rows = np.array(rows)
print(f"mean |rule - network| threshold: full {np.abs(rows[:, 1] - rows[:, 0]).mean():.3f}, ages {np.abs(rows[:, 2] - rows[:, 0]).mean():.3f}")
print("\nnudges: change in threshold per unit change of m2 and of m1 (m1 = 1.5, m2 = 1.0)")
for (i1, i2, t) in cases[:8]:
    base = np.array(thresholds(1.5, i1, 1.0, i2, t))
    d2 = (np.array(thresholds(1.5, i1, 1.05, i2, t)) - base) / 0.05
    d1 = (np.array(thresholds(1.55, i1, 1.0, i2, t)) - base) / 0.05
    print(f"  {i1} {i2} {t}: d/dm2 network {d2[0]:+.2f} rule {d2[1]:+.2f} ages {d2[2]:+.2f} | d/dm1 network {d1[0]:+.2f} rule {d1[1]:+.2f} ages {d1[2]:+.2f}")

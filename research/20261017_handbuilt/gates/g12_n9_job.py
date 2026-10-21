"""What does the late gate (neuron 9) do through its weights onto the hiding neurons (-0.69 onto 1, -0.64 onto 6,
-0.07 onto 8)? Removing that path drops lists whose second-largest is an overtaken record from 95.6% to about 49%
(g02_jobs.log). Here: what those lists name instead, which scores move, and how the hiding neurons' switch points
change one and two steps after a record.
Part 1: 200,000 random lists (seed 12). Part 2: lists with one overtaking: p = 1.0 at position i, the new largest
m = 1.0 + d at position r > i, every other number -0.5 (no other positive number), and the same with random fillers."""
import numpy as np
from gatelib import WI, WH, WO, CANC, MEM, lists, run, answer

X = lists(12); y = answer(X); L0, P0, H0 = run(X)
wh = WH.copy(); wh[CANC, 9] = 0
L1, P1, H1 = run(X, params=(WI, wh, WO))
o = X.argsort(1); iL, iS = o[:, -1], o[:, -2]
rev = iS < iL
ok0 = L0.argmax(1) == y; ok1 = L1.argmax(1) == y
a1 = L1.argmax(1)
print(f"revealed lists (second-largest came before the largest): intact {100 * ok0[rev].mean():.1f}%, path removed {100 * ok1[rev].mean():.1f}%")
bad = rev & ok0 & ~ok1
print(f"  of the revealed lists the removal breaks ({bad.sum()}), the network now names:")
print(f"    the largest's position     {100 * (a1[bad] == iL[bad]).mean():.1f}%")
print(f"    one after the largest      {100 * (a1[bad] == iL[bad] + 1).mean():.1f}%")
print(f"    the third-largest          {100 * (a1[bad] == o[bad, -3]).mean():.1f}%")
print(f"    one after the second       {100 * (a1[bad] == iS[bad] + 1).mean():.1f}%")
print(f"    a later position than L    {100 * (a1[bad] > iL[bad]).mean():.1f}%")
for gap in (1, 2, 3):
    m = rev & (iL - iS == gap)
    print(f"  revealed, largest {gap} step(s) after the second: intact {100 * ok0[m].mean():.1f}%, removed {100 * ok1[m].mean():.1f}%")
m = rev & (iL - iS >= 4)
print(f"  revealed, largest 4+ steps after the second: intact {100 * ok0[m].mean():.1f}%, removed {100 * ok1[m].mean():.1f}%")
for k in range(0, 10):
    m = rev & (iL == k)
    if m.sum():
        print(f"  revealed, largest at {k}: intact {100 * ok0[m].mean():.1f}%, removed {100 * ok1[m].mean():.1f}%  ({m.sum()} lists)")
kept = ~rev
print(f"kept lists (second after the largest): intact {100 * ok0[kept].mean():.1f}%, removed {100 * ok1[kept].mean():.1f}%")

print("\nPart 2: one overtaking, other numbers -0.5. Scores of S = p's position and L = the largest's position, and the best other.")
for d in (0.05, 0.2, 0.5):
    for i, r in ((1, 3), (2, 5), (3, 4), (4, 8), (0, 6)):
        x = np.full((1, 10), -0.5); x[0, i] = 1.0; x[0, r] = 1.0 + d
        s0 = run(x)[0][0]; s1 = run(x, params=(WI, wh, WO))[0][0]
        oth0 = np.delete(s0, [i, r]).max(); oth1 = np.delete(s1, [i, r]).max()
        print(f"  d={d:4.2f} p at {i}, m at {r}: intact S {s0[i]:6.1f} L {s0[r]:6.1f} other {oth0:6.1f} | removed S {s1[i]:6.1f} L {s1[r]:6.1f} other {oth1:6.1f}")

print("\nswitch points of the hiding neurons k steps after a record, as a share of the new largest m and the overtaken p")
print("(fit r = a p + b m on lists with p > 0.2 and m > p + 0.05 and negative numbers in between; intact / path removed)")
rng = np.random.default_rng(13); Z = rng.standard_normal((400000, 10))
for params, name in (((WI, WH, WO), "intact"), ((WI, wh, WO), "removed")):
    _, P, H = run(Z, params=params)
    for k in (1, 2, 3):
        rows = []
        for r in range(2, 9 - k):
            p = np.maximum(Z[:, :r].max(1), 0); ok = (Z[:, r] > p + 0.05) & (p > 0.2)
            if k > 1:
                ok &= (Z[:, r + 1:r + k] < 0).all(1)
            u = H[:, r + k - 1] @ params[1].T
            refs = np.stack([-u[:, j] / WI[j] for j in CANC], 1)
            rows.append(np.c_[p[ok], Z[ok, r], refs[ok]])
        R = np.concatenate(rows)
        out = []
        for c, j in enumerate(CANC):
            a, b = np.linalg.lstsq(R[:, :2], R[:, 2 + c], rcond=None)[0]
            out.append(f"{j}: {a:+.2f} p {b:+.2f} m")
        print(f"  {name:8s} k={k}: " + " | ".join(out))

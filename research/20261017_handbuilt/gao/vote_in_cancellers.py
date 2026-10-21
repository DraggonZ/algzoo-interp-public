"""Does the network store the vote of the number right after a record inside the cancelling neurons, so that the bigger
number that later silences them deletes it? Nudge that number z by 0.01 and split the change of its own position's
final score into the cancelling neurons' part and the rest. Groups: z one step after a record R (z positive, below R,
kept), compared with z three steps after R; each with the record still the largest at the end (no overtaking) and with
a bigger number arriving later (overtaking). 200,000 lists."""
import numpy as np
import simlib

X = simlib.lists(777, 200_000); B = len(X); C = [1, 6, 7, 8]; rest = [n for n in range(16) if n not in C]
_, _, H = simlib.run(X)
cm = np.maximum.accumulate(X, 1); prevmax = np.c_[np.full((B, 1), -np.inf), cm[:, :-1]]; rec = X > prevmax
import sys
for d in ((1, 3) if len(sys.argv) < 2 else tuple(int(v) for v in sys.argv[1:])):
    for over in (False, True):
        vals = {"cancelling": [], "rest": []}
        for p in range(1, 6):
            zp = p + d
            if zp > 7: continue
            R = X[:, p]; z = X[:, zp]
            m2 = np.sort(np.c_[np.zeros((B, 2)), X[:, :zp]], 1)[:, -2]
            ok = rec[:, p] & (cm[:, p] == R) & (z > 0) & (z < R) & (z >= 0.75 * m2) & (cm[:, zp] == R)
            bigger_later = cm[:, -1] > R
            ok &= bigger_later if over else ~bigger_later
            idx = np.flatnonzero(ok)[:4000]
            if len(idx) < 50: continue
            Xn = X[idx].copy(); Xn[:, zp] += 0.01; _, _, Hn = simlib.run(Xn)
            dh = (Hn[:, -1] - H[idx, -1]) / 0.01; w = simlib.WO[zp]
            vals["cancelling"] += list(dh[:, C] @ w[C]); vals["rest"] += list(dh[:, rest] @ w[rest])
        c, r = np.median(vals["cancelling"]), np.median(vals["rest"])
        print(f"z {d} step(s) after the record, {'overtaken later' if over else 'record stays largest'}: own-score gain per unit via cancelling neurons {c:6.1f}, via the rest {r:6.1f} ({len(vals['rest'])} lists)")

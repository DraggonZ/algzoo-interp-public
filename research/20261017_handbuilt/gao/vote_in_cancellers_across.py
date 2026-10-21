"""vote_in_cancellers_split.py on other trained networks: new111 (erases the number right after a record; cancelling
neurons 4, 8, 9, 12 by the 85% rule), new102 and new106 (erase two places after; [3, 5, 6] and [1, 2, 7, 8]).
z at distance d after a record (d = 1 for 111, 2 for 102 and 106), record still largest at the end vs overtaken;
z's own-score gain via the cancelling neurons vs the rest."""
from pathlib import Path
import numpy as np
from algzoo.core import load
import simlib

def net(k):
    return tuple(np.asarray(p, float) for p in load(str(next(Path("/data/output/research/algzoo/20261016_run3/models").glob(f"2nd_argmax_16_10_*_{k}.pth")))))
X = simlib.lists(777, 200_000); B = len(X)
cm = np.maximum.accumulate(X, 1); prevmax = np.c_[np.full((B, 1), -np.inf), cm[:, :-1]]; rec = X > prevmax
import sys
CASES = (("111", 1, [4, 8, 9, 12]), ("102", 2, [3, 5, 6]), ("106", 2, [1, 2, 7, 8])) if len(sys.argv) < 2 else (("111", 3, [4, 8, 9, 12]), ("102", 4, [3, 5, 6]), ("106", 4, [1, 2, 7, 8]))
for name, d, C in CASES:
    wi, wh, wo = net(name); wi = wi.reshape(-1); par = (wi, wh, wo); rest = [n for n in range(16) if n not in C]
    _, P, H = simlib.run(X, params=par)
    for over in (False, True):
        cs, rs = [], []
        for p in range(1, 7):
            zp = p + d
            if zp > 8: continue
            R = X[:, p]; z = X[:, zp]; m2 = np.sort(np.c_[np.zeros((B, 2)), X[:, :zp]], 1)[:, -2]
            ok = rec[:, p] & (cm[:, p] == R) & (z > 0) & (z < R) & (z >= 0.75 * m2) & (cm[:, zp] == R)
            ok &= (cm[:, -1] > R) if over else (cm[:, -1] == R)
            idx = np.flatnonzero(ok)[:4000]
            if len(idx) < 50: continue
            Xn = X[idx].copy(); Xn[:, zp] += 0.01; _, _, Hn = simlib.run(Xn, params=par)
            dh = (Hn[:, -1] - H[idx, -1]) / 0.01; w = wo[zp]
            cs += list(dh[:, C] @ w[C]); rs += list(dh[:, rest] @ w[rest])
        print(f"new{name}, z {d} after a record, {'overtaken later' if over else 'record stays largest'}: via cancelling neurons {np.median(cs):6.1f}, via the rest {np.median(rs):6.1f} ({len(cs)} lists)", flush=True)

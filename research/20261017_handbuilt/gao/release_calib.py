"""After an overtaking, do the overtaken record (second-largest) and a kept number that arrived after it (third-largest
candidate) carry the same vote per unit of value, so they compete by value? Lists: record R at position p, kept z at
p + d (d = 3 or 4, positive, below R), a bigger number later. Nudge R and z separately by 0.01; own-score gain per unit at
the end, total and via the cancelling neurons. Also before any overtaking (record stays largest) for z."""
import numpy as np
import simlib

X = simlib.lists(888, 300_000); B = len(X); C = [1, 6, 7, 8]
_, _, H = simlib.run(X)
cm = np.maximum.accumulate(X, 1); prevmax = np.c_[np.full((B, 1), -np.inf), cm[:, :-1]]; rec = X > prevmax
def gain(idx, pos):
    Xn = X[idx].copy(); Xn[np.arange(len(idx)), pos] += 0.01; _, _, Hn = simlib.run(Xn)
    dh = (Hn[:, -1] - H[idx, -1]) / 0.01; w = simlib.WO[pos]
    tot = (dh * w).sum(1); canc = (dh[:, C] * w[:, C]).sum(1); return tot, canc
for d in (3, 4):
    rows = []
    for p in range(0, 5):
        zp = p + d
        if zp > 7: continue
        R = X[:, p]; z = X[:, zp]; m2 = np.sort(np.c_[np.zeros((B, 2)), X[:, :zp]], 1)[:, -2]
        ok = rec[:, p] & (cm[:, zp] == R) & (z > 0) & (z < R) & (z >= 0.75 * m2) & (cm[:, -1] > R) & (cm[:, zp + 1:].max(1) > R if zp < 9 else False)
        idx = np.flatnonzero(ok)[:3000]
        if len(idx) < 50: continue
        tR, cR = gain(idx, np.full(len(idx), p)); tz, cz = gain(idx, np.full(len(idx), zp))
        rows.append((p, np.median(tR), np.median(tz), np.median(tz / np.where(np.abs(tR) > 1e-6, tR, np.nan)[~np.isnan(tR)] if False else tz) ))
        print(f"d={d}, record at {p}, kept number at {zp}, overtaken later ({len(idx)} lists): record's vote per unit {np.median(tR):5.1f}, kept number's {np.median(tz):5.1f}; ratio of medians {np.median(tz) / np.median(tR):.2f}", flush=True)

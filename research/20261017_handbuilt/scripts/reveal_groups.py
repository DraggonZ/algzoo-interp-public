"""Which neuron group makes the revealed vote sharp? Final scores split exactly by neuron group at the last step:
score = sum over groups of W_oh[:, G] h_T[G]. On random lists whose second-largest is an old largest (a record
later beaten), average each group's push (centred over positions 0-8) by offset from the old largest's position.
Compare with lists whose second-largest is a kept non-record."""
import numpy as np
from algzoo.core import example

wi, wh, wo = (np.asarray(w, dtype=float) for w in example()); wi = wi.reshape(-1)
groups = {"timer (0,3,9-15)": [0, 3, 9, 10, 11, 12, 13, 14, 15], "cancellers (1,6,7,8)": [1, 6, 7, 8],
          "store, detector (2,4)": [2, 4], "gate (5)": [5]}
rng = np.random.default_rng(23)
X = rng.standard_normal((1_000_000, 10)); B, n = X.shape
h = np.zeros((B, 16))
for t in range(n): h = np.maximum(h @ wh.T + X[:, t:t + 1] * wi, 0)
z = np.zeros((B, 2))
m1p = np.stack([np.max(np.c_[z, X[:, :t]], 1) for t in range(n)], 1)
record = X > m1p
o = X.argsort(1); sec, lar = o[:, -2], o[:, -1]
r = np.arange(B)
is_rev = record[r, sec] & (sec < lar) & (sec >= 2) & (sec <= 6)
is_kept = ~record[r, sec] & (sec >= 2) & (sec <= 6)
np.set_printoptions(precision=1, suppress=True, linewidth=150)
for label, sel in (("second = revealed old largest", is_rev), ("second = kept number", is_kept)):
    print(f"--- {label} ({sel.sum()} lists); push by offset -2..+2 from the second's position (own = offset 0)")
    total = np.zeros(5)
    for gname, G in groups.items():
        s = h[sel][:, G] @ wo[:, G].T
        s = s[:, :9] - s[:, :9].mean(1, keepdims=True)
        p = sec[sel]
        prof = np.array([s[np.arange(len(p)), p + d].mean() for d in (-2, -1, 0, 1, 2)])
        total += prof
        print(f"  {gname:24s} {prof}")
    print(f"  {'all':24s} {total}")

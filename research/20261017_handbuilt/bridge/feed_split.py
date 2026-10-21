"""Follow-up to feed_test.py. Split the memory's part of a hidden record's profile into the share written through the
detector (original minus 'no detector -> memory') and the share fed by the hiding neurons (original minus 'no hiding
-> memory'). For each share: fraction of its absolute size at positions up to the record (r and before) and after it.
Same split for v18 (detector -> memory, and clock and flip -> memory, which are its forward copies)."""
import numpy as np
from algzoo.core import example
from algzoo.handbuilt_v18 import build_v18
rng = np.random.default_rng(2); X = rng.standard_normal((600000, 10))
def final(P, X):
    wi, wh, wo = P; h = np.zeros((len(X), len(wi)))
    for t in range(10): h = np.maximum(h @ wh.T + X[:, t:t + 1] * wi, 0)
    return h
T = tuple(np.asarray(p, float) for p in example()); net = build_v18(); I = net.idx; V = tuple(np.asarray(p, float) for p in net.params)
nets = {"trained": (T, [0, 3, 10, 11, 12, 13, 14, 15], [1, 6, 7, 8], 4),
        "v18": (V, [I[f"mem{j}"] for j in range(8)], [I[k] for k in ("clock0", "clock1", "clock2", "flip")], I["detect"])}
eps = 1e-3
for name, (P, MEM, HID, DET) in nets.items():
    wi, wh, wo = P
    for r in (2, 4, 6):
        pre = np.maximum(X[:, :r].max(1), 0)
        ok = (X[:, r] > pre + 0.05) & (X[:, r] > 0.3) & (X[:, r + 1:].max(1) < X[:, r])
        Xs = X[ok]; Xp = Xs.copy(); Xp[:, r] += eps
        def prof(whv):
            Q = (wi, whv, wo); dh = (final(Q, Xp) - final(Q, Xs)) / eps
            return np.median(dh[:, MEM] @ wo[:, MEM].T, 0)[:9]
        base = prof(wh)
        w = wh.copy(); w[MEM, DET] = 0; det_part = base - prof(w)
        w = wh.copy(); w[np.ix_(MEM, HID)] = 0; hid_part = base - prof(w)
        upto = np.arange(9) <= r
        f = lambda v: np.abs(v[upto]).sum() / np.abs(v).sum()
        print(f"{name:7s} record at {r}: detector share at positions <= r {f(det_part):.2f}; hiding neurons' share at positions <= r {f(hid_part):.2f}"
              f"   (positions <= r are {upto.mean():.2f} of 0-8)")

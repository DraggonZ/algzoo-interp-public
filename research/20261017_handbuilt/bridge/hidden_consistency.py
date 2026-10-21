"""Sharper test of the consistency reading, on the hiding itself. Lists with a largest at r (2-5) that stays largest.
Leak per unit = change of its own score minus the median change elsewhere, when the largest is nudged. Variants of the
trained network at steps where the hiding group is partly off:
  as trained;
  consistent unclipping: the hiding neurons themselves keep their unclipped pre-activations (everything downstream,
    the recurrence and the readout see the same unclipped values);
  memory fed unclipped: the hiding neurons clip, but the memory receives their unclipped values;
  memory fed random-sign: the memory receives the clip amount with a random sign per list and neuron.
Prediction if only consistency matters for hiding: consistent unclipping leaks little; the two inconsistent ones more.
Also accuracy on 200,000 random lists."""
import numpy as np
from algzoo.core import example
wi, wh, wo = (np.asarray(p, float) for p in example())
HID = [1, 6, 7, 8]; MEM = [0, 3, 10, 11, 12, 13, 14, 15]
rng = np.random.default_rng(10); X = rng.standard_normal((600000, 10)); N0 = len(X)
SG = np.where(rng.random((N0, 4)) < 0.5, -1.0, 1.0)
def run(X, mode, sg):
    N = len(X); h = np.zeros((N, 16)); hp = h.copy()
    for t in range(10):
        pre = h @ wh.T + X[:, t:t + 1] * wi
        if mode == "memory fed unclipped": pre[:, MEM] += (hp[:, HID] - h[:, HID]) @ wh[np.ix_(MEM, HID)].T
        if mode == "memory fed random-sign": pre[:, MEM] += (sg * (hp[:, HID] - h[:, HID])) @ wh[np.ix_(MEM, HID)].T
        new = np.maximum(pre, 0); off = (new[:, HID] == 0).sum(1); part = (off > 0) & (off < 4)
        hp = new.copy(); hp[np.ix_(part, HID)] = pre[np.ix_(part, HID)]
        h = hp if mode == "consistent unclipping" else new
    return h @ wo.T
y = np.argsort(X[:200000], 1)[:, -2]
eps = 1e-3
for mode in ("as trained", "consistent unclipping", "memory fed unclipped", "memory fed random-sign"):
    acc = (run(X[:200000], mode, SG[:200000]).argmax(1) == y).mean(); out = []
    for r in (2, 3, 4, 5):
        pre = np.maximum(X[:, :r].max(1), 0); ok = (X[:, r] > pre + 0.05) & (X[:, r] > 0.3) & (X[:, r + 1:].max(1) < X[:, r])
        Xs = X[ok]; Xp = Xs.copy(); Xp[:, r] += eps
        d = (run(Xp, mode, SG[ok]) - run(Xs, mode, SG[ok])) / eps; leak = d[:, r] - np.median(np.delete(d, r, 1), 1)
        out.append(f"r{r} leak {np.median(leak):6.1f} (|leak| > 20 on {np.mean(np.abs(leak) > 20):.2f})")
    print(f"{mode:24s} accuracy {acc:.3f} | " + " | ".join(out))

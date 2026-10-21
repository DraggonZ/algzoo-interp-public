"""Question: does the store's timing signal re-drive the hiding neurons, so that a partial switch does not leave the
largest's hiding corrupted? Intervention: the hiding neurons' input from the store (neuron 2) uses k x the largest so
far instead of the store's value, with k the least-squares ratio store / largest on these lists (analysis of the trained
network; removes the store's slow wave of when the largest arrived, keeps its size). Compare: leak per unit of a
largest that stays largest, split by later partial switch (as hidden_after_clip.py); accuracy. Control: the same
replacement for the memory's input from the store instead."""
import numpy as np
from algzoo.core import example
wi, wh, wo = (np.asarray(p, float) for p in example())
HID = [1, 6, 7, 8]; MEM = [0, 3, 10, 11, 12, 13, 14, 15]
rng = np.random.default_rng(11); X = rng.standard_normal((600000, 10))
def run(X, target=None, k=0.0):
    N = len(X); h = np.zeros((N, 16)); m1 = np.zeros(N); parts = []
    for t in range(10):
        pre = h @ wh.T + X[:, t:t + 1] * wi
        if target is not None:
            pre[:, target] += np.outer(k * m1 - h[:, 2], wh[target, 2])
        h = np.maximum(pre, 0); m1 = np.maximum(m1, X[:, t])
        o = (h[:, HID] == 0).sum(1); parts.append((o > 0) & (o < 4))
    return h @ wo.T, np.stack(parts, 1)
# k: store / largest so far, least squares over steps 2-9
h = np.zeros((200000, 16)); num = den = 0.0; m1 = np.zeros(200000)
for t in range(10):
    h = np.maximum(h @ wh.T + X[:200000, t:t + 1] * wi, 0); m1 = np.maximum(m1, X[:200000, t])
    if t >= 2: num += (h[:, 2] * m1).sum(); den += (m1 * m1).sum()
k = num / den; print(f"store / largest so far: k = {k:.2f}")
y = np.argsort(X[:200000], 1)[:, -2]; eps = 1e-3
for name, tg in [("as trained", None), ("hiding neurons get k x largest", HID), ("memory gets k x largest (control)", MEM)]:
    acc = (run(X[:200000], tg, k)[0].argmax(1) == y).mean(); out = []
    for r in (2, 3, 4):
        pre = np.maximum(X[:, :r].max(1), 0); ok = (X[:, r] > pre + 0.05) & (X[:, r] > 0.3) & (X[:, r + 1:].max(1) < X[:, r])
        Xs = X[ok]; Xp = Xs.copy(); Xp[:, r] += eps
        L0, parts = run(Xs, tg, k); L1, _ = run(Xp, tg, k); d = (L1 - L0) / eps
        leak = d[:, r] - np.median(np.delete(d, r, 1), 1); cl = parts[:, r + 1:].any(1)
        out.append(f"r{r}: leak with partial {np.median(leak[cl]):6.1f}, without {np.median(leak[~cl]):6.1f}")
    print(f"{name:34s} accuracy {acc:.3f} | " + " | ".join(out))

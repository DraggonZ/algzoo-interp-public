"""Question: after a partial switch, does the current largest stay hidden? Lists where the largest arrives at step r
(2-5) and stays largest; split by whether its hiding group partly switched at some later step. Nudge the largest by
eps: change of the score at its own position minus the median change at the other positions (its leak per unit), median
over lists. Trained network; v18 at g = 10 (partly switches) and g = 10,000 (does not)."""
import numpy as np
from algzoo.core import example
from algzoo.handbuilt_v18 import build_v18
rng = np.random.default_rng(10); X = rng.standard_normal((600000, 10)); N = len(X)
def run(P, hid, X):
    P = [np.asarray(p, float) for p in P]; h = np.zeros((len(X), len(P[0]))); parts = []
    for t in range(10):
        h = np.maximum(h @ P[1].T + X[:, t:t + 1] * P[0], 0); o = (h[:, hid] == 0).sum(1); parts.append((o > 0) & (o < len(hid)))
    return h @ P[2].T, np.stack(parts, 1)
I = build_v18().idx; hid18 = [I[k] for k in ("clock0", "clock1", "clock2", "flip")]
nets = {"trained": (example(), [1, 6, 7, 8]), "v18 g 10": (build_v18(g=10, a=10, w=10, lam=10.0, c=-9.0, eps=0.1).params, hid18),
        "v18 g 10,000": (build_v18().params, hid18)}
eps = 1e-3
for name, (P, hid) in nets.items():
    out = []
    for r in (2, 3, 4, 5):
        pre = np.maximum(X[:, :r].max(1), 0)
        ok = (X[:, r] > pre + 0.05) & (X[:, r] > 0.3) & (X[:, r + 1:].max(1) < X[:, r])
        Xs = X[ok]; Xp = Xs.copy(); Xp[:, r] += eps
        L0, parts = run(P, hid, Xs); L1, _ = run(P, hid, Xp)
        d = (L1 - L0) / eps; leak = d[:, r] - np.median(np.delete(d, r, 1), 1)
        clipped = parts[:, r + 1:].any(1)
        out.append(f"r{r}: partial {clipped.mean():.2f}, leak with {np.median(leak[clipped]) if clipped.any() else float('nan'):6.2f}, without {np.median(leak[~clipped]):6.2f}")
    print(f"{name:13s} " + " | ".join(out))

"""Cancellation across the AlgZoo second-largest family: every hidden size d and list length n trained on 2^30
sequences, five seeds. Per network: accuracy and the median ratio of the winner's lead to the pushes on it
(1 = no cancellation). Per (d, n): median over seeds that reach at least 80% accuracy. 20,000 lists each."""
from pathlib import Path
import numpy as np
from algzoo.core import load

files = sorted(Path("/data/output/research/algzoo/models/zoo").glob("2nd_argmax_*_*_1073741824_*.pth"))
res = {}
for f in files:
    _, _, d, n, _, seed = f.stem.split("_"); d, n = int(d), int(n)
    wi, wh, wo = (np.asarray(p, float) for p in load(str(f))); wi = wi.reshape(-1)
    X = np.random.default_rng(7).standard_normal((20_000, n)); y = X.argsort(1)[:, -2]; r = np.arange(len(X))
    h = np.zeros((len(X), d))
    for t in range(n):
        h = np.maximum(h @ wh.T + X[:, t:t + 1] * wi, 0)
    L = h @ wo.T; k = L.argmax(1); j = np.argsort(L, 1)[:, -2]
    lead = L[r, k] - L[r, j]; push = np.abs((wo[k] - wo[j]) * h).sum(1)
    res.setdefault((d, n), []).append(((k == y).mean(), np.median(lead / np.maximum(push, 1e-12))))
ds = sorted({d for d, _ in res}); ns = sorted({n for _, n in res})
print("median lead / pushes over seeds with accuracy >= 80% (accuracy in brackets); rows: hidden size, columns: list length")
print("d\\n " + "".join(f"{n:>15d}" for n in ns))
for d in ds:
    row = []
    for n in ns:
        v = [x for x in res.get((d, n), []) if x[0] >= 0.8]
        row.append(f"{np.median([x[1] for x in v]):6.3f} ({np.median([x[0] for x in v]):.2f})" if v else f"{'-':>15s}")
    print(f"{d:3d} " + "".join(f"{c:>15s}" for c in row))

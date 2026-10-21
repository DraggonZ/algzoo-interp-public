"""Robustness across trained networks: accuracy with every neuron's true value plus Gaussian noise of s x its typical
active value (active steps only), s = 0.003, 0.01, 0.03; and the median ratio of the decision margin to the pushes on
the winner's lead. Compared with accuracy and the forgetting cutoff (forget_across.log). 50,000 lists."""
import re
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from algzoo.core import example, zoo, load
import simlib

cut = {m.group(1): float(m.group(2)) for m in re.finditer(r"  (\S+)\s+cutoff ([0-9.]+)", open("forget_across.log").read())}
models = {"ARC": example()}
for s in range(5):
    models[f"zoo{s}"] = zoo(16, 10, s)
for f in sorted(Path("/data/output/research/algzoo/20261016_run3/models").glob("2nd_argmax_16_10_*_1*.pth")):
    models["new" + f.stem.split("_")[-1]] = load(str(f))
X = simlib.lists(1414, 50_000); y = X.argsort(1)[:, -2]; rng = np.random.default_rng(0); rows = []
for name, w in models.items():
    P = tuple(np.asarray(p, float) for p in w); P = (P[0].reshape(-1), P[1], P[2])
    L, _, H = simlib.run(X, params=P); acc = (L.argmax(1) == y).mean()
    typ = np.array([np.median(H[:, :, n][H[:, :, n] > 0]) if (H[:, :, n] > 0).any() else 1.0 for n in range(16)])
    k = L.argmax(1); j = np.argsort(L, 1)[:, -2]; lead = L[np.arange(len(L)), k] - L[np.arange(len(L)), j]
    ratio = np.median(lead / np.maximum(np.abs((P[2][k] - P[2][j]) * H[:, 9]).sum(1), 1e-12))
    accs = []
    for s in (0.003, 0.01, 0.03):
        h = np.zeros((len(X), 16))
        for t in range(10):
            h = np.maximum(h @ P[1].T + X[:, t:t + 1] * P[0], 0); h = np.maximum(h + s * typ * rng.standard_normal(h.shape) * (h > 0), 0)
        accs.append(((h @ P[2].T).argmax(1) == y).mean())
    rows.append((name, acc, cut.get(name, np.nan), ratio, *accs))
    print(f"{name:7s} acc {acc:.3f} cutoff {cut.get(name, np.nan):.2f} margin/pushes {ratio:.3f}; with noise 0.003 {accs[0]:.3f}, 0.01 {accs[1]:.3f}, 0.03 {accs[2]:.3f}; loss at 0.01 {acc - accs[1]:.3f}", flush=True)
a = np.array([r[1] for r in rows]); c = np.array([r[2] for r in rows]); rt = np.array([r[3] for r in rows]); loss = a - np.array([r[5] for r in rows])
print(f"rank correlations over {len(rows)} networks: accuracy vs loss at 1% noise {spearmanr(a, loss)[0]:.2f} (p {spearmanr(a, loss)[1]:.3f}); "
      f"accuracy vs margin/pushes {spearmanr(a, rt)[0]:.2f} (p {spearmanr(a, rt)[1]:.3f}); forgetting cutoff vs loss {spearmanr(c, loss)[0]:.2f} (p {spearmanr(c, loss)[1]:.3f})")

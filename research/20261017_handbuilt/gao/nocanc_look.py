"""The five trained networks with no neuron silenced (zero) by every new largest (gao/canc_across.log): for each neuron,
its mean activity at steps where a new largest arrives relative to other steps, and the share of record steps where it is
zero. Neurons turned well down at a new largest but not to zero would be 'partial' cancelling neurons."""
from pathlib import Path
import numpy as np
from algzoo.core import zoo, load
import simlib
from simlib import lists, task_vars
nets = {"zoo1": zoo(16, 10, 1), "zoo4": zoo(16, 10, 4)}
for k in ("110", "112", "114"):
    nets["new" + k] = load(str(next(Path("/data/output/research/algzoo/20261016_run3/models").glob(f"2nd_argmax_16_10_*_{k}.pth"))))
X = lists(202, 30_000); V = task_vars(X); T = np.arange(10)[None, :]
rec = (V["newmax"] > 0) & (T >= 2); oth = (V["newmax"] == 0) & (T >= 2)
for name, w in nets.items():
    P = tuple(np.asarray(p, float) for p in w); P = (P[0].reshape(-1), P[1], P[2]); _, _, H = simlib.run(X, params=P)
    rows = []
    for n in range(16):
        a, b = H[:, :, n][rec].mean(), H[:, :, n][oth].mean()
        if b > 0.5:
            rows.append((a / b, (H[:, :, n][rec] <= 1e-9).mean(), n))
    rows.sort()
    print(name, "lowest activity at a new largest relative to other steps:", ", ".join(f"n{n} {r:.2f} (zero {z:.0%})" for r, z, n in rows[:5]))

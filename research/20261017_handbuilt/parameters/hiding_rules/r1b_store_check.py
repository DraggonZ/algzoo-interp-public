"""r1b diagnostic: does the store/detector 2x2 recursion (common.sd_rule) match ARC's store (2) and detector (4)?
And which other sources feed the store and detector (ARC's weights times typical activity)?"""
import sys
sys.path.insert(0, "rules")
import numpy as np
from common import WI, WH, sd_rule, task_vars, lag
import simlib

X = simlib.lists(101, 50_000); _, P, H = simlib.run(X)
V = task_vars(X); sd = sd_rule(X)
Hp = np.concatenate([np.zeros((len(X), 1, 16)), H[:, :-1]], 1)
J = lag(V["jump"])
for i, n in enumerate((2, 4)):
    err = sd[..., i] - Hp[..., n]
    print(f"neuron {n}: rms of recursion {np.sqrt((err ** 2).mean()):.3f} against activity rms {np.sqrt((Hp[..., n] ** 2).mean()):.3f}; "
          f"error vs previous jump slope {np.polyfit(J.ravel(), err.ravel(), 1)[0]:+.2f}")
    contrib = {k: np.sqrt(((WH[n, k] * H[..., k]) ** 2).mean()) for k in range(16) if k not in (2, 4)}
    print("   other sources' rms contribution: " + ", ".join(f"{k}:{v:.2f}" for k, v in sorted(contrib.items(), key=lambda z: -z[1])[:6]))

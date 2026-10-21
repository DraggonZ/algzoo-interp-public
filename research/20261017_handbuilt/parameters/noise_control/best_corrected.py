"""The best rule with round 14's arrival choice removed: r15a's contender push with the extra step's arrival 0
(r12a's) instead of the list's minimum, and the 5 structure bits of that choice refunded. On ARC, v2c."""
import sys
from pathlib import Path
import numpy as np
import torch as th
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pricer  # noqa: E402
from run_searches import Var, push_dq  # noqa: E402

P = Path(__file__).resolve().parent.parent
S = pricer.setup("arc", 0)
r15 = pricer.load(P / "open/rules/r15a_contender.py", "r15a_bc")
x, h = None, None
g = np.random.default_rng(101); x = g.standard_normal((400_000, 10))
h = np.zeros((400_000, 16))
for t in range(10):
    h = np.maximum(h @ S.WH.T + x[:, t, None] * S.WI[None, :], 0)
h11 = np.maximum(h @ S.WH.T, 0)                       # arrival 0
rank = (-x).argsort(1).argsort(1)
REG = np.stack([np.linalg.solve(h[rank[:, k] < 3].T @ h[rank[:, k] < 3], h[rank[:, k] < 3].T @ (h11[rank[:, k] < 3] @ S.WO[k - 1]))
                for k in range(1, 9)])
v = Var(r15, dq=push_dq(r15, REG, r15._REG)); v.STRUCTURE_BITS = r15.STRUCTURE_BITS - 5
for lab, rule in [("r15a as merged (minimum arrival)", r15), ("r15a with arrival 0, 5 bits refunded", v)]:
    a = pricer.price(S, rule, "own")[0]; b = pricer.price(S, rule, "exact")[0]; c = pricer.price(S, rule, "bits")[0]
    print(f"{lab}: own fit {a:+.1f}, exact fit {b:+.1f}, bit-optimal gain {c:+.1f}")

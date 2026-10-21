"""r15a look 8: forms of 'contender' with a fresh arrival: (a) top 3 of x_0..x_9; (b) second or third only (the
largest is never the answer); (c) top 3 of the list after the step (x_1..x_9 and the newcomer); (d) top 3 at both
times. Seed from FIT_SEED."""
import os
import numpy as np
from r15a_var import *

rng = np.random.default_rng(int(os.environ.get("FIT_SEED", "101")) + 1000)
u = rng.standard_normal(N)
h11 = step(H, u)
rank = (-XS).argsort(1).argsort(1)
L2 = np.concatenate([XS[:, 1:], u[:, None]], 1)
rank2 = (-L2).argsort(1).argsort(1)          # position k at h_10 is position k-1 in L2
wa = np.stack([(rank[:, k] < 3) for k in PUSH_ROWS]).astype(float)
wb = np.stack([(rank[:, k] >= 1) & (rank[:, k] < 3) for k in PUSH_ROWS]).astype(float)
wc = np.stack([(rank2[:, k - 1] < 3) for k in PUSH_ROWS]).astype(float)
wd = wa * wc
for lab, w in [("(a) top3 of x_0..x_9", wa), ("(b) 2nd or 3rd", wb), ("(c) top3 after the step", wc),
               ("(d) top3 at both times", wd)]:
    both(lab + ", fresh", push(h11, w), None, 0)

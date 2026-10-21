"""r15a look 6: seeds (FIT_SEED env) for the contender push; arrivals that cannot disturb the contenders."""
import os
import numpy as np
from r15a_var import *

print("FIT_SEED", os.environ.get("FIT_SEED", "101"))
rank = (-XS).argsort(1).argsort(1)
w3 = np.stack([(rank[:, k] < 3).astype(float) for k in PUSH_ROWS])
row0 = np.tile(-WO[0] / 9, (8, 1))
both("r14a (min arrival, offset)", push(step(H, XS.min(1))), row0, 8)
both("top3, min arrival", push(step(H, XS.min(1)), w3), None, 5)
rng = np.random.default_rng(int(os.environ.get("FIT_SEED", "101")) + 1000)
both("top3, fresh arrival", push(step(H, rng.standard_normal(N)), w3), None, 0)
# fresh arrival truncated below the list's third-largest (cannot change which numbers are the top three)
third = -np.sort(-XS, 1)[:, 2]
u = rng.standard_normal(N)
bad = u >= third
while bad.any():
    u[bad] = rng.standard_normal(bad.sum()); bad = u >= third
both("top3, fresh below the third-largest", push(step(H, u), w3), None, 0)

"""r15a look 5: how the contender restriction depends on its cut (top 3..10 by the position's rank in its list),
on a soft form (weight = 1 for top 3, then 0), on the arrival, and on the list seed (via a second seed's states).
No offset (r13b's), which is worth nothing under v2c (look 4)."""
import numpy as np
from r15a_var import *

rank = (-XS).argsort(1).argsort(1)      # 0 = largest
h11m = step(H, XS.min(1))
for top in [3, 4, 5, 6, 8, 10]:
    w = np.stack([(rank[:, k] < top).astype(float) for k in PUSH_ROWS])
    both(f"top{top}, min arrival", push(h11m, w), None, 5)
# rank of the arriving position's number relative to the contenders: weight by rank of row k-1's number at h_11?
# the number at position k at h_10 is the number at position k-1 at h_11: same number, same rank.
w3 = np.stack([(rank[:, k] < 3).astype(float) for k in PUSH_ROWS])
both("top3, arrival 0", push(step(H, np.zeros(N)), w3), None, 0)
both("top3, fresh arrival", push(step(H, np.random.default_rng(5).standard_normal(N)), w3), None, 0)
both("top3, arrival -1", push(step(H, -np.ones(N)), w3), None, 0)
# top 3 among the 9 numbers that stay in the list after x_0 leaves? (rank of x_k among x_1..x_9)
rank9 = (-XS[:, 1:]).argsort(1).argsort(1)
w39 = np.stack([(rank9[:, k - 1] < 3).astype(float) for k in PUSH_ROWS])
both("top3 among x_1..x_9, min arrival", push(h11m, w39), None, 5)

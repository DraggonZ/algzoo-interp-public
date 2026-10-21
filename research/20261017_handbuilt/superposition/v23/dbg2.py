import numpy as np
from algzoo.handbuilt_v23 import build_v23, scores
np.set_printoptions(precision=3, suppress=True, linewidth=180)
net = build_v23()
base = np.full(10, 0.01)
for i, j in ((0, 1), (2, 5), (5, 6), (3, 8), (7, 9), (8, 9), (1, 4)):
    x = base.copy(); x[i] = 0.5; x[j] = 0.9
    print((i, j), scores(net, x[None])[0])

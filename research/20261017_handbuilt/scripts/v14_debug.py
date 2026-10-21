import numpy as np
from handbuilt_v9 import build_v7, scores
np.set_printoptions(precision=3, suppress=True, linewidth=160)
net = build_v7(clock="jump", beta=0.0, mem_jump=True, w=0.0, lam_s=10.0, keep_rec=True, hide=1.0)
for vals in ({2: 1.0}, {2: 1.0, 6: 1.5}, {2: 1.0, 4: 0.8, 6: 1.5}, {2: 1.0, 6: 1.5, 7: 0.8}, {1: 0.5, 3: 1.0, 6: 1.5}):
    x = np.full(10, -0.5)
    for k, v in vals.items(): x[k] = v
    s = scores(net, x[None])[0]; print(vals, s - s.mean())

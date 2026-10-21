import numpy as np
from algzoo.handbuilt_v23 import build_v23, scores
np.set_printoptions(precision=3, suppress=True, linewidth=180)
net = build_v23(w=0.0)
for x in ([0.3, 0.2, 0.1, 0.25, 0.15, 0.05, 0.12, 0.22, 0.18, 0.08], [0.9, 0.2, 0.1, 0.25, 0.15, 0.05, 0.12, 0.22, 0.18, 0.08],
          [0.2, 0.3, 0.1, 0.25, 0.15, 0.05, 0.12, 0.22, 0.18, 0.08], [0.2, 0.1, 0.1, 0.25, 0.15, 0.9, 0.12, 0.22, 0.18, 0.08]):
    X = np.array([x]); print(np.array(x)); print(scores(net, X)[0], " intended", np.array(x) * 1.0)

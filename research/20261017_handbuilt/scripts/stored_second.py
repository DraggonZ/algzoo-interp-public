"""How strongly do neurons 7 and 8 (which feed neuron 5) hold the second-largest so far, depending on when it arrived
after the largest? Lists: largest M at 0, second S at pS, zeros elsewhere (no later numbers). At each later step t,
fit the activity of neurons 7 and 8 as a * M + b * S over a grid of M in [1.5, 3] and S in [0.4, 1.2] (least squares,
exact within one on/off pattern) and report b (weight on the second) and a, by pS and t."""
import numpy as np

from algzoo.core import example

wi, wh, wo = example()
Ms, Ss = np.meshgrid(np.linspace(1.5, 3.0, 7), np.linspace(0.4, 1.2, 7)); Ms, Ss = Ms.ravel(), Ss.ravel()
for k in (7, 8):
    print(f"neuron {k}: weight on the second-largest (b) and on the largest (a), by when the second arrived (rows) and the step (columns)")
    for pS in range(1, 8):
        X = np.zeros((len(Ms), 10)); X[:, 0] = Ms; X[:, pS] = Ss
        h = np.zeros((len(X), 16)); row = []
        for t in range(10):
            h = np.maximum(0, h @ wh.T + X[:, t:t + 1] * wi)
            if t > pS:
                A = np.column_stack([Ms, Ss]); coef, res, *_ = np.linalg.lstsq(A, h[:, k], rcond=None)
                r2 = 1 - (res[0] / np.sum((h[:, k] - h[:, k].mean()) ** 2) if len(res) and np.var(h[:, k]) > 0 else 0)
                row.append(f"t{t}: b {coef[1]:+5.1f} a {coef[0]:+5.1f}")
        print(f"  second at {pS}: " + " | ".join(row))

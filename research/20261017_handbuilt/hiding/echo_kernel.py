"""Echo of a clip: if hiding neuron i is clipped at step j by e (its value is 0 instead of its negative pre-activation),
the network carries the difference e on. Its effect on each hiding neuron's pre-activation lag steps later, per unit
of e: measured by adding a small amount to neuron i's value at step j in the running network (median over lists and
steps j = 2..6), against the four's own block alone (W_cc^lag). If they agree, the echo stays inside the four."""
import numpy as np
from common import WI, WH, C, lists, run
X = lists(5, 20_000); Wcc = WH[np.ix_(C, C)]; eps = 1e-4
_, P0, H0 = run(X)
for i, ni in enumerate(C):
    print(f"== unit added to neuron {ni}'s value")
    for lag in range(1, 5):
        meas = []
        for j in range(2, 10 - lag):
            def hook(t, pre, hn, hprev, j=j):
                if t == j: hn = hn.copy(); hn[:, ni] += eps
                return hn
            _, P1, _ = run(X, hook=hook); meas.append((P1[:, j + lag, C] - P0[:, j + lag, C]) / eps)
        meas = np.concatenate(meas); own = np.linalg.matrix_power(Wcc, lag)[:, i]
        print(f"   lag {lag}: whole network (median) " + " ".join(f"{v:6.3f}" for v in np.median(meas, 0)) +
              f" | four alone " + " ".join(f"{v:6.3f}" for v in own) + f" | spread (IQR, median) {np.median(np.subtract(*np.percentile(meas, [75, 25], 0))):.3f}")

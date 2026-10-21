"""Where does the trained network erase the vote of y (right after record P at r) once P is overtaken at L?
Twin runs y vs y + eps; median sensitivity d h_t / d y per neuron and step, overtaken (x_L = P + 0.5) vs not
(x_L = -0.5). r = 3, y at 4, L = 6; other numbers random below 0.5 P."""
import numpy as np
from algzoo.core import example
from algzoo.handbuilt import run
np.set_printoptions(precision=1, suppress=True, linewidth=170)
params = example(); wo = params[2]
rng = np.random.default_rng(0); n = 20000
X = rng.uniform(-1.5, 0.45, (n, 10)); P = rng.uniform(0.9, 1.6, n); X[:, 3] = P; X[:, 4] = P * rng.uniform(0.6, 0.9, n)
X[:, :3] = np.minimum(X[:, :3], 0.7 * P[:, None])
eps = 0.01
for name, xl in (("stays", -0.5), ("overtaken", None)):
    Z = X.copy(); Z[:, 6] = P + 0.5 if xl is None else xl
    Z2 = Z.copy(); Z2[:, 4] += eps
    H = run(params, Z); H2 = run(params, Z2)
    dH = np.median((H2 - H) / eps, 0)          # [11, 16]
    print(name, "| final d score_4 / dy:", np.median(((H2[:, -1] - H[:, -1]) @ wo.T)[:, 4] / eps).round(1))
    for t in range(4, 11): print(f"  t={t - 1}", dH[t])
    on = np.mean(H[:, :, :] > 0, 0)
    print("  share on, neurons 0,3,10-15 at t=6..9:", on[7:11][:, [0, 3, 10, 11, 12, 13, 14, 15]].round(2).tolist())

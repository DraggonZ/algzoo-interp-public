"""ARC's neuron 5, one step after an arrival: fit its pre-activation as
a*x_t + b*second_so_far + c*largest_so_far (+ next input), on random inputs."""
import numpy as np

from algzoo.core import example

w_hi, w_hh, w_oh = example()
rng = np.random.default_rng(6)
x = rng.standard_normal((200000, 10))
H = np.zeros((len(x), 11, 16))
for t in range(10):
    H[:, t + 1] = np.maximum(H[:, t] @ w_hh.T + x[:, t, None] * w_hi, 0)
rows, ys = [], []
for t in range(2, 9):
    pre = np.sort(np.concatenate([np.zeros((len(x), 2)), x[:, :t]], 1), 1)
    m1, m2 = pre[:, -1], pre[:, -2]
    pre5 = H[:, t + 1] @ w_hh[5] + x[:, t + 1] * w_hi[5]  # neuron 5 at the step after x_t
    ok = (x[:, t] > 0) & (x[:, t] < m1)
    rows.append(np.c_[x[ok, t], m2[ok], m1[ok], x[ok, t + 1]]); ys.append(pre5[ok])
A, y = np.concatenate(rows), np.concatenate(ys)
coef, *_ = np.linalg.lstsq(A, y, rcond=None)
r2 = 1 - ((A @ coef - y) ** 2).mean() / y.var()
print("neuron 5 pre-activation ~ %.2f x_t %+.2f second %+.2f largest %+.2f next input   (R^2 %.3f)" % (*coef, r2))
print("implied switch point x_t / second (largest term ignored): %.2f" % (-coef[1] / coef[0]))
print("neuron 5 input weight %.2f; weights from lowering neurons 1,6,7,8: %s; from 2,4: %s" % (w_hi[5], np.round(w_hh[5, [1, 6, 7, 8]], 2), np.round(w_hh[5, [2, 4]], 2)))

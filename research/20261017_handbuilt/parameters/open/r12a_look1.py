"""r12a look 1: ceilings on the current best, and the readout push residual's structure.
(a) value of halving / zeroing each readout row's error (at refitted gain);
(b) value of readout row 0 exact (one new row);
(c) push residual e_k = ARC row k - g push(row k-1) on the push columns: cosine with candidate directions."""
import numpy as np
import torch as th
import r12a_x as X

R = X.load(X.BEST)
net0, d0 = X.price(R)
print(f"base {net0:+.1f}")
W = d0["w"]
S = X.S


def arc(n):
    return S.w_flat[S.IDX[n]]


# (a) shrink errors of one readout row
for k in range(1, 10):
    for f in (0.5, 0.0):
        ex = {n: arc(n) + f * (W[n] - arc(n)) for n in W if n.endswith(f"->out{k}")}
        net, d = X.price(X.Extend(R, ex))
        print(f"row out{k} error x{f}: {net - net0:+.1f}  (gain {d['C'][0]:.3f})")
# (b) row 0 exact / at 20% noise
rng = np.random.default_rng(0)
for f in (0.0, 0.1, 0.2, 0.3):
    ex = {f"{i}->out0": X.WO[0, i] * (1 + f * rng.standard_normal()) for i in range(16)}
    net, d = X.price(X.Extend(R, ex))
    print(f"row out0 at {f} relative noise: {net - net0:+.1f}")

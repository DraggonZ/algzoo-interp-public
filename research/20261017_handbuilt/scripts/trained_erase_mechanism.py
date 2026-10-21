"""ARC's model: how does a below-second number's vote vanish one step later,
while a new second's vote lasts? Track d h_s / d x_t for s = t, t+1, t+2 and its
visible part (score at own slot minus mean), split by neuron group and by kind of
arrival. Only arrivals with x_t > 0 at t <= 7, and next input x_{t+1} below the
current second (a quiet step), so that the next step is comparable."""
import numpy as np
import torch as th

from algzoo.core import example

w_hi, w_hh, w_oh = example()
rng = np.random.default_rng(3)
x = rng.standard_normal((40000, 10))
B, n = x.shape
H = np.zeros((B, n + 1, 16))
for t in range(n):
    H[:, t + 1] = np.maximum(H[:, t] @ w_hh.T + x[:, t, None] * w_hi, 0)
arr = np.zeros((B, n), int)
for t in range(n):
    arr[:, t] = (x[:, :t] > x[:, t:t + 1]).sum(1)
pm2 = np.array([[np.sort(np.r_[0, 0, x[b, :t]])[-2] for t in range(n)] for b in range(2000)])  # unused
groups = {"lowering 1,6,7,8": [1, 6, 7, 8], "max/detector 2,4": [2, 4], "other ten": [0, 3, 5, 9, 10, 11, 12, 13, 14, 15]}
Wd = w_oh - w_oh.mean(0)
for kind, code in (("new second", 1), ("below second", 2)):
    print(f"\n== {kind} arrivals ==")
    for t in (3, 5):
        m = (arr[:, t] >= code) if code == 2 else (arr[:, t] == 1)
        m &= (x[:, t] > 0) & (arr[:, t + 1] >= 2)
        idx = np.where(m)[0][:3000]
        slot = t
        for lag in (0, 1, 2):
            s = t + lag
            # derivative of h_{s+1} wrt x_t, via active masks
            d = (H[idx, t + 1] > 0) * w_hi
            for u in range(t + 1, s + 1):
                d = (H[idx, u + 1] > 0) * (d @ w_hh.T)
            vis_total = (d @ Wd.T)[:, slot - lag + lag]  # score change at x_t's slot (if readout now)
            parts = {g: (d[:, ix] @ Wd[:, ix].T)[:, slot] for g, ix in groups.items()}
            print(f" t={t} lag {lag}: |dh| {np.median(np.linalg.norm(d,axis=1)):6.1f}  own-slot vote {np.median(vis_total):7.1f}  by group: "
                  + ", ".join(f"{g} {np.median(v):7.1f}" for g, v in parts.items())
                  + "  | lowering active: " + str((H[idx, s + 1][:, [1, 6, 7, 8]] > 0).mean(0).round(2)))

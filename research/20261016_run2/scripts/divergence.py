"""Where does a noisy copy's two-spike trajectory first depart from the base model's? (deterministic θ grid)

For every position pair and grid angle where the base model is right and the copy is wrong at the end,
find the first (step, neuron) at which the copy's ReLU on/off pattern differs from the base's.
Report the share of lost mass by neuron, by phase (at a spike vs during the empty steps after the last spike),
and by age of the last spike.
"""
import numpy as np
import torch as th

from algzoo.bench import load_bench
from algzoo.core import example
from algzoo.exact_z2 import HI, LO, g_angle

dev = "cuda"
NG = 1000
tg = LO + (np.arange(NG) + 0.5) * (HI - LO) / NG
W = th.as_tensor(g_angle(tg) * (HI - LO) / NG, device=dev)
c, s = th.as_tensor(np.cos(tg), device=dev), th.as_tensor(np.sin(tg), device=dev)


@th.no_grad()
def patterns(params, p1, p2):
    w_hi, w_hh, w_oh = (th.as_tensor(p, device=dev, dtype=th.float64) for p in params)
    h = th.zeros(NG, 16, device=dev, dtype=th.float64)
    pats = []
    for t in range(10):
        x = c if t == p1 else (s if t == p2 else th.zeros_like(c))
        z = h @ w_hh.T + x[:, None] * w_hi
        pats.append(z > 0)
        h = th.relu(z)
    return th.stack(pats, 1), (h @ w_oh.T).argmax(-1) == p2       # [NG, 10, 16], [NG]


base = example()
names, acc, models = load_bench(("mult",))
C = [1, 2, 4, 6, 7, 8]
tot_neuron = np.zeros(16)
tot_phase = np.zeros(2)
tot_lost = 0.0
for m in models:
    for p1 in range(10):
        for p2 in range(10):
            if p1 == p2:
                continue
            T = max(p1, p2)
            Pb, okb = patterns(base, p1, p2)
            Pc, okc = patterns(m, p1, p2)
            lost = okb & ~okc
            if not lost.any():
                continue
            diff = (Pb != Pc)[lost]                     # [L, 10, 16]
            w = W[lost]
            anyd = diff.any(-1)                          # [L, 10]
            first_t = th.where(anyd.any(-1), anyd.float().argmax(-1), th.full_like(w, 10, dtype=th.long))
            for k in range(diff.shape[0]):
                t = first_t[k].item()
                if t == 10:
                    continue
                nz = th.nonzero(diff[k, t]).flatten().cpu().numpy()
                tot_neuron[nz] += w[k].item() / len(nz)
                tot_phase[int(t > T)] += w[k].item()
            tot_lost += w.sum().item()
print(f"lost mass (base right, copy wrong), summed over 96 copies x 90 pairs, per-pair units: {tot_lost:.2f}")
print("share of lost mass whose first pattern difference is at or before the last spike / after it:", (tot_phase / tot_phase.sum()).round(3))
order = np.argsort(-tot_neuron)
print("first-differing neuron, share of lost mass:", ", ".join(f"n{j}{'*' if j in C else ''} {tot_neuron[j] / tot_neuron.sum():.2f}" for j in order[:10]))

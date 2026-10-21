"""Does an explanation reproduce the trained network's mistakes? On the grid (40 x 48 x 48, no sampling), for each
replacement: overall agreement; among lists where the trained network is wrong, the share where the replaced network
gives the same wrong answer ('mistakes reproduced'); among lists where the trained network is right, the share where
the replaced network is right too. Baselines: a perfect solver (always the true second-largest) and the first twin
(gains 10, matched), which works differently inside."""
import sys
import numpy as np, torch
sys.path.insert(0, "../propagate")
import store_ladder as sl, store_ladder2 as s2, detector_ladder as dl, pair_ladder as pl
DEV = "cuda"; dt = torch.float64
d = np.load("../../bridge/v18ts_matched2.npz"); H = [torch.tensor(d[k], device=DEV, dtype=dt) for k in ("wi", "wh", "wo")]
def twin(X):
    h = torch.zeros(len(X), 16, device=DEV, dtype=dt)
    for t in range(10): h = torch.relu(h @ H[1].T + X[:, t:t + 1] * H[0])
    return h @ H[2].T
# mean-ablation controls: the neuron held at its average value at each step (over 300,000 random lists)
_X = torch.randn(300000, 10, device=DEV, dtype=dt, generator=torch.Generator(device=DEV).manual_seed(31)); _h = torch.zeros(len(_X), 16, device=DEV, dtype=dt); MEANS = []
for t in range(10): _h = torch.relu(_h @ sl.wh.T + _X[:, t:t + 1] * sl.wi); MEANS.append(_h.mean(0))
def mean_ablate(j):
    def f(X):
        h = torch.zeros(len(X), 16, device=DEV, dtype=dt)
        for t in range(10): h = torch.relu(h @ sl.wh.T + X[:, t:t + 1] * sl.wi); h[:, j] = MEANS[t][j]
        return h @ sl.wo.T
    return f
models = [("perfect solver (always right)", None),
          ("control: store held at its average", mean_ablate(2)),
          ("control: detector held at its average", mean_ablate(4)),
          ("first twin (works differently inside)", twin),
          ("store as 17.6 x the largest", lambda X: sl.run(X, rule=lambda L, a: 17.6 * L)),
          ("store as age table", lambda X: sl.run(X, rule=lambda L, a: sl.r[a.clamp(max=9)] * L)),
          ("store as age table + residual", s2.run4),
          ("detector fires at a new largest", lambda X: dl.run(X, dl.rules[0][1])),
          ("detector own weights + residual", lambda X: dl.run(X, dl.d4)),
          ("store + detector, both with residual", lambda X: pl.run(X, "residual", dl.d4))]
acc = {k: np.zeros(3) for k, _ in models}; Zw = Zr = Zall = 0.0
nV = len(sl.wtt)
for i in range(0, 720, 16):
    tt = sl.full_mean.T[i:i + 16]; pos = tt.repeat_interleave(nV, 0); val = sl.Vt.repeat(len(tt), 1); n = len(pos)
    bmu = -torch.exp(-val[:, 2] ** 2 / 2) / np.sqrt(2 * np.pi) / (0.5 * (1 + torch.erf(val[:, 2] / np.sqrt(2))))
    X = bmu[:, None].repeat(1, 10); ar = torch.arange(n, device=DEV)
    for k in range(3): X[ar, pos[:, k]] = val[:, k]
    w = sl.wtt.repeat(len(tt)); y = pos[:, 1]; a0 = sl.run(X).argmax(1); wrong = a0 != y
    Zw += (w * wrong).sum().item(); Zr += (w * ~wrong).sum().item(); Zall += w.sum().item()
    for name, f in models:
        a1 = y if f is None else f(X).argmax(1)
        acc[name] += [(w * (a1 == a0)).sum().item(), (w * (wrong & (a1 == a0))).sum().item(), (w * (~wrong & (a1 == y))).sum().item()]
print(f"trained network wrong on {100*Zw/Zall:.2f}% of the grid's probability")
print(f"{'':42s} {'agreement':>10s} {'mistakes reproduced':>20s} {'right where trained is right':>29s}")
for name, (a, mw, rr) in acc.items():
    print(f"{name:42s} {100*a/Zall:9.2f}% {100*mw/Zw:19.1f}% {100*rr/Zr:28.2f}%")

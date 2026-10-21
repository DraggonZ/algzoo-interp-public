"""Is the 30% of ARC's mistakes shared by other trained networks generic to near ties? A perfect second-largest solver
that reads each number with an error, at a size that gives about ARC's accuracy, scored on ARC's mistakes (grid
40 x 48 x 48, no sampling over lists; 8 draws of the reading errors). Error kinds:
  fresh:    every number read with + N(0, s^2), new each list (an imprecise but unbiased reader)
  relative: every number read as x (1 + N(0, s^2))
  offsets:  a fixed offset per position, N(0, s^2), the same for all lists (one draw = one 'network')."""
import numpy as np, torch
import store_ladder as sl
DEV = "cuda"; dt = torch.float64
KINDS = {"fresh": [0.02, 0.05, 0.1, 0.2], "relative": [0.02, 0.05, 0.1, 0.2], "offsets": [0.02, 0.05, 0.1, 0.2]}
ND = 8
if __name__ == "__main__":
    g = torch.Generator(device=DEV).manual_seed(3)
    OFF = {s: torch.randn(ND, 10, device=DEV, dtype=dt, generator=g) * s for s in KINDS["offsets"]}
    nV = len(sl.wtt); keys = [(k, s) for k, ss in KINDS.items() for s in ss]
    rep = {key: 0.0 for key in keys}; acc = {key: 0.0 for key in keys}; Zw = Zall = 0.0
    for i in range(0, 720, 16):
        tt = sl.full_mean.T[i:i + 16]; pos = tt.repeat_interleave(nV, 0); val = sl.Vt.repeat(len(tt), 1); n = len(pos)
        bmu = -torch.exp(-val[:, 2] ** 2 / 2) / np.sqrt(2 * np.pi) / (0.5 * (1 + torch.erf(val[:, 2] / np.sqrt(2))))
        X = bmu[:, None].repeat(1, 10); ar = torch.arange(n, device=DEV)
        for k in range(3): X[ar, pos[:, k]] = val[:, k]
        w = sl.wtt.repeat(len(tt)); y = pos[:, 1]; a0 = sl.run(X).argmax(1); wr = a0 != y
        Zw += (w * wr).sum().item(); Zall += w.sum().item()
        for (k, s) in keys:
            for d in range(ND):
                if k == "fresh": Xn = X + s * torch.randn(X.shape, device=DEV, dtype=dt, generator=g)
                elif k == "relative": Xn = X * (1 + s * torch.randn(X.shape, device=DEV, dtype=dt, generator=g))
                else: Xn = X + OFF[s][d]
                a1 = Xn.topk(2, 1).indices[:, 1]
                rep[(k, s)] += (w * (wr & (a1 == a0))).sum().item() / ND; acc[(k, s)] += (w * (a1 == y)).sum().item() / ND
    print(f"ARC's network wrong on {100*Zw/Zall:.2f}% of the grid (accuracy {100-100*Zw/Zall:.2f}%)")
    print(f"{'error kind':10s} {'size':>5s} {'accuracy':>9s} {'ARC mistakes reproduced':>24s}")
    for key in keys: print(f"{key[0]:10s} {key[1]:5.2f} {100*acc[key]/Zall:8.2f}% {100*rep[key]/Zw:23.1f}%")

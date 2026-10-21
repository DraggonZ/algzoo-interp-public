"""Ceiling for any explanation that uses only the three largest numbers: on the grid lists where ARC errs (the other
seven at their mean below the third), redraw the other seven from the normal truncated below the third (8 draws) and
ask how often ARC makes the same wrong choice. Also ARC's accuracy with redrawn smaller numbers on those lists, and the
same for one other trained network (209) against itself."""
import sys, numpy as np, torch
sys.path.insert(0, "../hilton/topdown"); sys.path.insert(0, "../hilton/propagate")
import store_ladder as sl
from common import *
if __name__ == "__main__":
    nV = len(sl.wtt); g = torch.Generator(device=DEV).manual_seed(9); res = {}
    for name, W in (("ARC", ARC), ("trained 209", trained(209))):
        st = np.zeros(4)
        for i in range(0, 720, 16):
            tt = sl.full_mean.T[i:i + 16]; pos = tt.repeat_interleave(nV, 0); val = sl.Vt.repeat(len(tt), 1); n = len(pos)
            bmu = -torch.exp(-val[:, 2] ** 2 / 2) / np.sqrt(2 * np.pi) / (0.5 * (1 + torch.erf(val[:, 2] / np.sqrt(2))))
            X = bmu[:, None].repeat(1, 10); ar = torch.arange(n, device=DEV)
            for k in range(3): X[ar, pos[:, k]] = val[:, k]
            w = sl.wtt.repeat(len(tt)); y = pos[:, 1]; a0 = scores(W, X).argmax(1); wr = a0 != y
            top = torch.zeros(n, 10, dtype=torch.bool, device=DEV)
            for k in range(3): top[ar, pos[:, k]] = True
            c3 = 0.5 * (1 + torch.erf(val[:, 2] / np.sqrt(2)))
            for d in range(8):
                u = torch.rand(n, 10, device=DEV, dtype=dt, generator=g) * c3[:, None]
                Xr = torch.where(top, X, np.sqrt(2) * torch.erfinv((2 * u - 1).clamp(-1 + 1e-12, 1 - 1e-12)))
                a1 = scores(W, Xr).argmax(1)
                st += np.array([(w * wr * (a1 == a0)).sum().item(), (w * wr * (a1 == y)).sum().item(), (w * ~wr * (a1 == y)).sum().item(), 0]) / 8
            st[3] += 0; res.setdefault(name, [0, 0]); res[name][0] += (w * wr).sum().item(); res[name][1] += (w * ~wr).sum().item()
        Zw, Zr = res[name]
        print(f"{name}: on its own grid mistakes, with the smaller numbers redrawn: same wrong choice {100*st[0]/Zw:.1f}%, right {100*st[1]/Zw:.1f}%; on lists it gets right, still right {100*st[2]/Zr:.1f}%")

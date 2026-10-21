"""The asymmetry comes from a few weak (second, largest) cells (table.log): largest at step 9 (mean 92%) and largest at 0
with a late second (second 7: 87%, 9: 91%). For each weak cell and two strong ones: what the network names instead, and
which rotation mode's 20% removal at the final state fixes it (the 126° mode fixed the slot-0 case). Prediction
(written first): the same mode balance explains all the weak cells."""
import sys; sys.path.insert(0, "../steer"); sys.path.insert(0, "../extreme")
import numpy as np
from harness import *
from modes import ROT, R, Linv, ang
groups = {}
for i in np.argsort(ang): groups.setdefault(int(round(ang[i])), []).append(i)
g = torch.Generator(device=DEV).manual_seed(12); N = 100_000
def cell(ps, pl):
    s = torch.randn(N, device=DEV, dtype=dt, generator=g).abs() + 0.5
    l = s * (1 + 0.3 * torch.randn(N, device=DEV, dtype=dt, generator=g).abs()) + 1e-3
    X = s[:, None] * 0.97 * torch.rand(N, 10, device=DEV, dtype=dt, generator=g)
    X = torch.where(torch.rand(N, 10, device=DEV, dtype=dt, generator=g) < 0.3, -torch.randn(N, 10, device=DEV, dtype=dt, generator=g).abs(), X)
    X[:, ps] = s; X[:, pl] = l; return X
def final(X):
    _, Hs = run(ARC, X, keep=True); return Hs[-1]
for ps, pl in ((0, 9), (6, 9), (7, 9), (2, 9), (7, 0), (9, 0), (3, 4), (5, 1)):
    X = cell(ps, pl); h = final(X); a = (h @ ARC[2].T).argmax(1); wr = a != ps
    named = np.bincount(a[wr].cpu().numpy(), minlength=10) / max(int(wr.sum()), 1)
    top = ", ".join(f"{k}:{100*named[k]:.0f}%" for k in np.argsort(-named)[:3])
    row = []
    for key, sel in sorted(groups.items()):
        P = torch.tensor((R[:, sel] @ Linv[sel, :]).real, device=DEV, dtype=dt); h2 = h.clone(); h2[:, ROT] = h[:, ROT] - 0.2 * h[:, ROT] @ P.T
        row.append(f"{key}°: {100*((h2 @ ARC[2].T).argmax(1) != ps).double().mean().item():5.1f}")
    print(f"second {ps}, largest {pl}: wrong {100*wr.double().mean().item():5.1f}%, names {top} | wrong after removing 20% of a mode: " + " ".join(row), flush=True)

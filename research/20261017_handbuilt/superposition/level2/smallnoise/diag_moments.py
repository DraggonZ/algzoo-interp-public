"""Development (after the frozen evaluation): which weights-only way of getting the leads' moments over the six numbers
below the fourth is right on networks where the Jacobian at their mean fails (seeds 114, 207) and on ARC?
Per condition, third > 0, truth from 20,000 sampled backgrounds (diagnosis only)."""
import sys, numpy as np, torch, models
from lib import *
for name in sys.argv[1:]:
    W = load_net(name); rng = np.random.default_rng(1); X0 = rng.standard_normal((4000, 10)); o = np.argsort(-X0, 1)[:, :3]; v = np.take_along_axis(X0, o, 1)
    keep = v[:, 2] > 0; o, v = o[keep][:300], v[keep][:300]; n = len(o); pos = torch.tensor(o, device=DEV); val = torch.tensor(v, device=DEV, dtype=dt)
    tr = np.zeros(n)
    for i in range(n):
        p = ncdf(val[i, 2]).item(); X = nicdf((torch.rand(20000, 10, device=DEV, dtype=dt) * p).clamp_min(1e-300)); X[:, pos[i]] = val[i]
        tr[i] = (run(W, X).argmax(1) == pos[i, 1]).double().mean().item()
    print(f"net {name}: truth {tr.mean():.4f}")
    for lab, f in [("Jacobian at the mean (frozen)", lambda: models.model_fourth(W, pos, val, 4)),
                   ("axis, 3 nodes", lambda: models.model_fourth_m(W, pos, val, 4, "axis", 3)),
                   ("axis, 6 nodes", lambda: models.model_fourth_m(W, pos, val, 4, "axis", 6)),
                   ("cubature 16 points", lambda: models.model_fourth_m(W, pos, val, 4, "qmc", M=16)),
                   ("cubature 64 points", lambda: models.model_fourth_m(W, pos, val, 4, "qmc", M=64))]:
        p = f()[0].cpu().numpy(); e = p - tr
        print(f"  {lab:30s} bias {e.mean():+.4f}  mean abs error {np.abs(e).mean():.4f}", flush=True)

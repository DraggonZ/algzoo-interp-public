"""Development: lists whose third is negative, per condition (truth 20,000 sampled backgrounds; diagnosis only).
Three or nine leads, moments from the Jacobian at the mean (frozen) or from exact runs along each axis."""
import sys, numpy as np, torch, models
from lib import *
SIGN = int(sys.argv[1])
for name in sys.argv[2:]:
    W = load_net(name); rng = np.random.default_rng(2); X0 = rng.standard_normal((40000, 10)); o = np.argsort(-X0, 1)[:, :3]; v = np.take_along_axis(X0, o, 1)
    keep = (v[:, 2] < 0) if SIGN < 0 else (v[:, 2] > 0); o, v = o[keep][:300], v[keep][:300]; n = len(o); pos = torch.tensor(o, device=DEV); val = torch.tensor(v, device=DEV, dtype=dt)
    tr = np.zeros((n, 4))
    for i in range(n):
        p = ncdf(val[i, 2]).item(); X = nicdf((torch.rand(20000, 10, device=DEV, dtype=dt) * p).clamp_min(1e-300)); X[:, pos[i]] = val[i]
        a = run(W, X).argmax(1).cpu().numpy(); tr[i] = [(a == o[i, 1]).mean(), (a == o[i, 0]).mean(), (a == o[i, 2]).mean(), (~np.isin(a, o[i])).mean()]
    print(f"net {name}, third {'<' if SIGN < 0 else '>'} 0: truth {tr[:,0].mean():.4f}, wrong naming largest/third/another {tr[:,1].mean():.4f}/{tr[:,2].mean():.4f}/{tr[:,3].mean():.4f}")
    for lab, f in [("Jacobian, 3 leads", lambda: models.model_fourth(W, pos, val, 4)), ("Jacobian, 9 leads (frozen)", lambda: models.model_fourth(W, pos, val, 4, True)),
                   ("axis, 3 leads", lambda: models.model_fourth_m(W, pos, val, 4, "axis", 3)), ("axis, 9 leads", lambda: models.model_fourth_m(W, pos, val, 4, "axis", 3, inc_small=True)),
                   ("axis, 3 leads + 1-D others, Qa=3", lambda: models.model_fourth_m(W, pos, val, 4, "axis", 3, small1d=True)),
                   ("axis, 3 leads + 1-D others, Qa=6", lambda: models.model_fourth_m(W, pos, val, 4, "axis", 6, small1d=True))]:
        p, sp = (t.cpu().numpy() for t in f()); e = p - tr[:, 0]
        print(f"  {lab:28s} bias {e.mean():+.4f} mean abs error {np.abs(e).mean():.4f} | wrong {sp[:,0].mean():.4f}/{sp[:,1].mean():.4f}/{sp[:,2].mean():.4f}", flush=True)

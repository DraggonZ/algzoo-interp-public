"""Lists whose third-largest is negative (5.5% of lists, ARC 83% right there, most 'another' mistakes): per-condition
check of the fourth-largest model with three leads and with all nine leads. Truth by sampling (diagnosis only)."""
import numpy as np, torch, models
from lib import *
W = load_net("ARC"); rng = np.random.default_rng(8); K = 20000
out = []
while len(out) < 400:
    X0 = rng.standard_normal((50000, 10)); o = np.argsort(-X0, 1)[:, :3]; v = np.take_along_axis(X0, o, 1); m = v[:, 2] < 0
    out += list(zip(o[m], v[m]))
o, v = map(np.array, zip(*out[:400])); pos = torch.tensor(o, device=DEV); val = torch.tensor(v, device=DEV, dtype=dt)
tr = np.zeros((400, 4))
for i in range(400):
    p = ncdf(val[i, 2]).item(); X = nicdf((torch.rand(K, 10, device=DEV, dtype=dt) * p).clamp_min(1e-300)); X[:, pos[i]] = val[i]
    a = run(W, X).argmax(1).cpu().numpy(); tr[i] = [(a == o[i, 1]).mean(), (a == o[i, 0]).mean(), (a == o[i, 2]).mean(), (~np.isin(a, o[i])).mean()]
print(f"third < 0: truth P(right) {tr[:,0].mean():.4f}; wrong to largest {tr[:,1].mean():.4f}, third {tr[:,2].mean():.4f}, smaller {tr[:,3].mean():.4f}")
for name, f in [("two", lambda: models.model_two(W, pos, val)), ("fourth, 3 leads", lambda: models.model_fourth(W, pos, val, 8)),
                ("fourth, 9 leads", lambda: models.model_fourth(W, pos, val, 8, True))]:
    pr, sp = (t.cpu().numpy() for t in f())
    print(f"  {name:16s} P(right) {pr.mean():.4f} bias {pr.mean()-tr[:,0].mean():+.4f} mean abs error {np.abs(pr-tr[:,0]).mean():.4f} | wrong to largest {sp[:,0].mean():.4f}, third {sp[:,1].mean():.4f}, smaller {sp[:,2].mean():.4f}")

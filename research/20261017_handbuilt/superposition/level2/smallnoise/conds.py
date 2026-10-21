"""Fixed diagnosis set: 400 random and 400 near-tie top-three conditions (ARC's network), truth P(right) from 20,000
sampled backgrounds each, split by what wins when wrong. Saved to conds_ARC.npz. Sampling for diagnosis only."""
import sys, numpy as np, torch
from lib import *
from diag import conditions, W
name = sys.argv[1] if len(sys.argv) > 1 else "ARC"
if name != "ARC":
    import diag; diag.W = W = load_net(name)
torch.manual_seed(5); K = 20000; res = {}
for label, near in [("random", False), ("near", True)]:
    o, v = conditions(400, near); n = len(o); tr = np.zeros((n, 4))
    for i in range(n):
        p = ncdf(torch.tensor(v[i, 2])).item(); X = nicdf((torch.rand(K, 10, device=DEV, dtype=dt) * p).clamp_min(1e-300))
        X[:, o[i]] = torch.tensor(v[i], device=DEV, dtype=dt); a = run(W, X).argmax(1).cpu().numpy()
        tr[i] = [(a == o[i, 1]).mean(), (a == o[i, 0]).mean(), (a == o[i, 2]).mean(), (~np.isin(a, o[i])).mean()]
    res[label + "_pos"] = o; res[label + "_val"] = v; res[label + "_truth"] = tr
    print(f"{label}: P(right) {tr[:,0].mean():.4f}; wrong naming largest {tr[:,1].mean():.4f}, third {tr[:,2].mean():.4f}, smaller {tr[:,3].mean():.4f}")
np.savez(f"conds_{name}.npz", **res)

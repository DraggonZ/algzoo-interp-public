"""Check the propagation at fixed top-three numbers against the network run on sampled background numbers
(sampling here only to check). 300 conditions taken from random lists."""
import numpy as np, torch
from scipy.stats import truncnorm
from adf import propagate, orthant, net, DEV
W = net(None); wi, wh, wo = (w.cpu().numpy() for w in W)
rng = np.random.default_rng(3); n = 300; K = 4000
X0 = rng.standard_normal((n, 10)); order = np.argsort(-X0, 1)[:, :3]; vals = np.take_along_axis(X0, order, 1)
pos = torch.tensor(order, device=DEV); val = torch.tensor(vals, device=DEV)
res = {}
for name, kw in [("full", {}), ("no cross-cov", {"cross": "zero"}), ("background mean only", {"bg_var": False})]:
    sm, sc = propagate(W, pos, val, **kw); res[name] = orthant(sm, sc, pos[:, 1])[0].cpu().numpy()
truth = np.zeros(n)
for i in range(n):
    X = truncnorm.rvs(-np.inf, vals[i, 2], size=(K, 10), random_state=rng)
    X[:, order[i]] = vals[i]
    h = np.zeros((K, 16))
    for t in range(10): h = np.maximum(h @ wh.T + X[:, t:t + 1] * wi, 0)
    truth[i] = ((h @ wo.T).argmax(1) == order[i, 1]).mean()
print(f"truth: mean P(right | top three) {truth.mean():.4f}  (share of conditions with P < 0.5: {(truth < .5).mean():.3f})")
for k, v in res.items():
    print(f"{k:22s} mean {v.mean():.4f}  mean abs error {np.abs(v - truth).mean():.4f}  corr {np.corrcoef(v, truth)[0,1]:.3f}  worst {np.abs(v-truth).max():.3f}")
bad = np.argsort(-np.abs(res["full"] - truth))[:8]
for i in bad: print("  worst:", order[i], vals[i].round(2), "truth", truth[i].round(3), "full", res["full"][i].round(3), "mean-only", res["background mean only"][i].round(3))
np.savez("check_conditional.npz", truth=truth, order=order, vals=vals, **{k.replace(" ", "_"): v for k, v in res.items()})

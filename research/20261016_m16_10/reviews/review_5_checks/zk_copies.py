"""Review 5 check F: Z3 with a negative background on the 96 noisy copies (diagnostic, sampled).
Backgrounds: 0; -0.5; each non-top-3 input set to E[X | X < third] = -phi(m3)/Phi(m3) (no fitting)."""
import numpy as np, torch
from scipy.stats import norm
from algzoo.bench import load_bench
from algzoo.core import example, zoo
names, acc, models = load_bench()
dev = "cuda"
rng = np.random.default_rng(31); N = 1 << 19
X = rng.standard_normal((N, 10)); o = np.argsort(X, 1); p2 = torch.tensor(o[:, -2], device=dev)
m3 = np.take_along_axis(X, o[:, -3:-2], 1)
cm = -norm.pdf(m3) / norm.cdf(m3)
def keep(k, bg):
    Xk = np.broadcast_to(bg, X.shape).copy() if np.ndim(bg) else np.full_like(X, bg)
    idx = o[:, -k:]; np.put_along_axis(Xk, idx, np.take_along_axis(X, idx, 1), 1)
    return torch.tensor(Xk, device=dev, dtype=torch.float32)
inputs = {"Z3 bg0": keep(3, 0.0), "Z3 bg-0.5": keep(3, -0.5), "Z3 bg cond.mean": keep(3, cm), "Z4 bg cond.mean": keep(4, np.take_along_axis(-norm.pdf(X) / norm.cdf(X), o[:, -4:-3], 1) * 0 + (-norm.pdf(np.take_along_axis(X, o[:, -4:-3], 1)) / norm.cdf(np.take_along_axis(X, o[:, -4:-3], 1))))}
def acc_of(params, Xt):
    w_hi, w_hh, w_oh = (torch.tensor(np.asarray(p), device=dev, dtype=torch.float32) for p in params)
    h = torch.zeros(len(Xt), 16, device=dev)
    for t in range(10):
        h = torch.relu(h @ w_hh.T + Xt[:, t, None] * w_hi)
    return ((h @ w_oh.T).argmax(1) == p2).float().mean().item()
sig = np.array([float(str(n).split("_")[0][4:]) for n in names])
res = {k: np.array([acc_of(m, v) for m in models]) for k, v in inputs.items()}
for k, v in res.items():
    e = v - acc
    per = " / ".join(f"{100*np.sqrt(np.mean(e[sig == s]**2)):.1f}" for s in [0.005, 0.01, 0.02])
    ok = acc >= 0.6
    print(f"{k}: RMSE vs true {100*np.sqrt(np.mean(e**2)):.1f} points (bias {100*e.mean():+.1f}); by sigma {per}; copies >=60%: {100*np.sqrt(np.mean(e[ok]**2)):.1f}")
print("trained models, cond.mean background:")
for name in ["example", 0, 1, 2, 3, 4]:
    params = example() if name == "example" else zoo(16, 10, name)
    full = acc_of(params, torch.tensor(X, device=dev, dtype=torch.float32))
    print(f"  {name}: full {full:.3f}, " + ", ".join(f"{k} {acc_of(params, v):.3f}" for k, v in inputs.items()))

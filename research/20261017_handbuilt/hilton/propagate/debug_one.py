"""One condition where the full propagation fails: compare its per-step neuron means and SDs with sampled ones."""
import numpy as np, torch
from scipy.stats import truncnorm
import adf
W = adf.net(None); wi, wh, wo = (w.cpu().numpy() for w in W)
order = np.array([[9, 2, 4]]); vals = np.array([[1.25, 1.2, 1.07]])
rng = np.random.default_rng(0); K = 20000
X = truncnorm.rvs(-np.inf, vals[0, 2], size=(K, 10), random_state=rng); X[:, order[0]] = vals[0]
pos = torch.tensor(order, device="cuda"); val = torch.tensor(vals, device="cuda")
bmu, bvar = adf.background(val[:, 2]); mu = torch.zeros(1, 16, device="cuda", dtype=torch.float64); cov = torch.zeros(1, 16, 16, device="cuda", dtype=torch.float64)
h = np.zeros((K, 16))
for t in range(10):
    xm = bmu.clone(); xv = bvar.clone()
    for k in range(3):
        hit = pos[:, k] == t; xm = torch.where(hit, val[:, k], xm); xv = torch.where(hit, torch.zeros_like(xv), xv)
    am = mu @ W[1].T + xm[:, None] * W[0]; ac = W[1] @ cov @ W[1].T + xv[:, None, None] * (W[0][:, None] * W[0][None, :])
    mu, cov = adf.relu_moments(am, ac)
    h = np.maximum(h @ wh.T + X[:, t:t + 1] * wi, 0)
    sd = torch.diagonal(cov[0]).sqrt().cpu().numpy()
    print(f"step {t}: worst mean gap " + "  ".join(f"n{j}:{mu[0,j].item():.2f}/{h[:,j].mean():.2f} sd {sd[j]:.2f}/{h[:,j].std():.2f}" for j in np.argsort(-np.abs(mu[0].cpu().numpy() - h.mean(0)))[:3]))
S = h @ wo.T; sm = mu @ W[2].T; sc = W[2] @ cov @ W[2].T
lead = S[:, 2:3] - S
print("sampled leads mean", lead.mean(0).round(1), "sd", lead.std(0).round(1))
A = np.eye(10); L = np.zeros((10, 10)); L[:, 2] = 1; L -= np.eye(10)
lm = (L @ sm[0].cpu().numpy()); lc = L @ sc[0].cpu().numpy() @ L.T
print("propagated mean  ", lm.round(1), "sd", np.sqrt(np.clip(np.diag(lc), 0, None)).round(1))

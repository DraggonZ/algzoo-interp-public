"""Development check: per competitor, the Gaussian-with-true-moments tail P(margin < 0) against the sampled share,
summed over nodes (positions of the largest and second-largest; values in 25 bins). Competitors: the largest, the
third-largest, and other positions by their position (0-9)."""
import numpy as np, torch
from scipy.stats import norm
from algzoo.core import example
DEV = "cuda"; dt = torch.float64
wi, wh, wo = [torch.as_tensor(np.asarray(w, float), device=DEV, dtype=dt) for w in example()]
B = 25; edges = torch.tensor(norm.ppf(np.linspace(0, 1, B + 1)[1:-1] * 0.5 + 0.5) - 0.3, device=DEV, dtype=dt)
nK = 100 * B * B
n_ = torch.zeros(nK, device=DEV, dtype=dt); s1 = torch.zeros(nK, 10, device=DEV, dtype=dt); s2 = s1.clone(); neg = s1.clone()
for sd in range(5):
    X = torch.randn(2_000_000, 10, device=DEV, dtype=dt, generator=torch.Generator(device=DEV).manual_seed(300 + sd))
    h = torch.zeros(len(X), 16, device=DEV, dtype=dt)
    for t in range(10): h = torch.relu(h @ wh.T + X[:, t:t + 1] * wi)
    sc = h @ wo.T; o = X.argsort(1, descending=True); pL, pS = o[:, 0], o[:, 1]
    iL = torch.bucketize(X.gather(1, pL[:, None])[:, 0], edges); iS = torch.bucketize(X.gather(1, pS[:, None])[:, 0], edges)
    c = ((pL * 10 + pS) * B + iL) * B + iS
    d = sc.gather(1, pS[:, None]) - sc
    n_.index_add_(0, c, torch.ones(len(X), device=DEV, dtype=dt)); s1.index_add_(0, c, d); s2.index_add_(0, c, d * d); neg.index_add_(0, c, (d < 0).to(dt))
    if sd == 0: pT0 = o[:, 2]
ok = n_ > 30; N = n_[ok].sum()
m = s1[ok] / n_[ok, None]; v = (s2[ok] / n_[ok, None] - m ** 2).clamp_min(1e-12)
gt = torch.special.ndtr(-m / v.sqrt()) * n_[ok, None] / N; st = neg[ok] / N
idx = torch.nonzero(ok)[:, 0]; pSn = (idx // (B * B)) % 10; pLn = idx // (B * B * 10)
gt[torch.arange(len(idx)), pSn] = 0; st[torch.arange(len(idx)), pSn] = 0
gL = gt[torch.arange(len(idx)), pLn].sum().item(); sL = st[torch.arange(len(idx)), pLn].sum().item()
print(f"competitor = the largest: Gaussian tail {100*gL:.2f} points, sampled {100*sL:.2f}")
gt[torch.arange(len(idx)), pLn] = 0; st[torch.arange(len(idx)), pLn] = 0
for k in range(10): print(f"competitor = other number at position {k}: Gaussian tail {100*gt[:, k].sum().item():.2f} points, sampled {100*st[:, k].sum().item():.2f}")
print(f"all others: Gaussian {100*gt.sum().item():.2f}, sampled {100*st.sum().item():.2f}")

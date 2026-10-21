"""Development check: is the 'one Gaussian per node' form itself too crude? Per node (positions of the largest and
second-largest; their values in 10 bins), the scores' true mean and covariance from sampled lists; the Gaussian win
probability from these true moments, against the sampled accuracy in that node. If the Gaussian with true moments
already falls short, the shape (not the propagation) is the limit. Also with nodes refined by the third-largest's
position (before or after the second)."""
import numpy as np, torch
from scipy.stats import norm, qmc
from algzoo.core import example
DEV = "cuda"; dt = torch.float64
wi, wh, wo = [torch.as_tensor(np.asarray(w, float), device=DEV, dtype=dt) for w in example()]
Z = torch.tensor(norm.ppf(np.clip(qmc.Sobol(9, scramble=True, seed=0).random(4096), 1e-12, 1 - 1e-12)), device=DEV, dtype=dt)
import sys
B = int(sys.argv[1]); edges = torch.tensor(norm.ppf(np.linspace(0, 1, B + 1)[1:-1] * 0.5 + 0.5) - 0.3, device=DEV, dtype=dt)
for refine in sys.argv[2:]:
    R = {'none': 1, 'Torder': 2, 'Tgap': 4, 'Tboth': 8}[refine]; nK = 10 * 10 * B * B * R
    n_ = torch.zeros(nK, device=DEV, dtype=dt); r_ = n_.clone(); s1 = torch.zeros(nK, 10, device=DEV, dtype=dt); s2 = torch.zeros(nK, 10, 10, device=DEV, dtype=dt)
    for sd in range(5):
        X = torch.randn(2_000_000, 10, device=DEV, dtype=dt, generator=torch.Generator(device=DEV).manual_seed(200 + sd))
        h = torch.zeros(len(X), 16, device=DEV, dtype=dt)
        for t in range(10): h = torch.relu(h @ wh.T + X[:, t:t + 1] * wi)
        sc = h @ wo.T; o = X.argsort(1, descending=True); pL, pS, pT = o[:, 0], o[:, 1], o[:, 2]
        iL = torch.bucketize(X.gather(1, pL[:, None])[:, 0], edges); iS = torch.bucketize(X.gather(1, pS[:, None])[:, 0], edges)
        c = ((pL * 10 + pS) * B + iL) * B + iS
        Tv = X.gather(1, pT[:, None])[:, 0]; Sv = X.gather(1, pS[:, None])[:, 0]; gb = torch.bucketize(Sv - Tv, torch.tensor([0.05, 0.15, 0.4], device=DEV, dtype=dt))
        if refine == 'Torder': c = c * 2 + (pT > pS).long()
        if refine == 'Tgap': c = c * 4 + gb
        if refine == 'Tboth': c = c * 8 + gb * 2 + (pT > pS).long()
        n_.index_add_(0, c, torch.ones(len(X), device=DEV, dtype=dt)); r_.index_add_(0, c, (sc.argmax(1) == pS).to(dt))
        # scores relative to the answer's score: d_k = s_pS - s_k
        d = sc.gather(1, pS[:, None]) - sc
        s1.index_add_(0, c, d); s2.index_add_(0, c, d[:, :, None] * d[:, None, :])
    ok = n_ > 30; idx = torch.nonzero(ok)[:, 0]
    m = s1[idx] / n_[idx, None]; C = s2[idx] / n_[idx, None, None] - m[:, :, None] * m[:, None, :]
    pSn = (idx // (B * B * R)) % 10
    keep = torch.ones(len(idx), 10, dtype=torch.bool, device=DEV); keep[torch.arange(len(idx)), pSn] = False
    m9 = m[keep].view(-1, 9); C9 = C[keep[:, :, None] & keep[:, None, :]].view(-1, 9, 9)
    ev, V = torch.linalg.eigh(C9); Lc = V * ev.clamp_min(1e-12).sqrt()[:, None, :]
    pw = torch.cat([((m9[a:a + 2000, None] + torch.einsum("qj,nij->nqi", Z, Lc[a:a + 2000])) > 0).all(2).to(dt).mean(1) for a in range(0, len(m9), 2000)])
    w = n_[idx] / n_.sum()
    print(f"bins {B}, refined by {refine}: nodes {len(idx)} (covering {100*w.sum().item():.2f}% of lists); sampled accuracy {100*(r_[idx].sum()/n_[idx].sum()).item():.2f}%, Gaussian with true moments {100*(w*pw).sum().item()/w.sum().item():.2f}%")

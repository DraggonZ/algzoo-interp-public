"""Development check for a held-out failure (seed from argv): per step and neuron, the propagated state's per-node
mean and SD against sampled lists in the same node (positions and grid cells of the running largest and second;
look-ahead sub-regimes pooled)."""
import sys, numpy as np, torch
from algzoo.core import load, example
from prop4 import Prop, DEV, dt
s = sys.argv[1]; G = int(sys.argv[2])
W = [np.asarray(w, float) for w in (example() if s == "ARC" else load(f"/data/output/research/algzoo/20261016_run3/models/2nd_argmax_16_10_1073741824_{s}.pth"))]
P = Prop(W, G=G, lookahead="auto", chunk=1_000_000, exact_last=False); P._choose_lookahead(10); P.run(1)   # pick look-ahead on the full run, then start
print("look-ahead", P.la)
X = torch.randn(2_000_000, 10, device=DEV, dtype=dt, generator=torch.Generator(device=DEV).manual_seed(7)); n = len(X)
edges = torch.tensor(P.edges[1:-1], device=DEV, dtype=dt)
wi, wh = P.wi, P.wh
h = torch.relu(X[:, :1] * wi); L = X[:, 0].clone(); S = torch.full((n,), -1e9, device=DEV, dtype=dt)
pL = torch.zeros(n, dtype=torch.long, device=DEV); pS = torch.full((n,), -1, dtype=torch.long, device=DEV)
for t in range(1, 10):
    x = X[:, t]; h = torch.relu(h @ wh.T + x[:, None] * wi)
    rec = x > L; sec = ~rec & (x > S)
    S = torch.where(rec, L, torch.where(sec, x, S)); pS = torch.where(rec, pL, torch.where(sec, t, pS))
    L = torch.where(rec, x, L); pL = torch.where(rec, t, pL)
    P.step(t)
    nb = P.nb; Wt = P.W.view(110, nb, -1).sum(1); M1 = P.M1.view(110, nb, -1, 16).sum(1)
    d2 = P.M2.view(110, nb, -1, 136).sum(1)[..., [i * 16 - i * (i - 1) // 2 for i in range(16)]]   # diagonal entries of the packed triangle
    iL = torch.bucketize(L, edges); iS = torch.where(S < -1e8, 0, torch.bucketize(S, edges) + 1)
    reg = pL * 11 + pS + 1; q = P._pk(iL, iS); w = Wt[reg, q]; ok = w > 0
    mu = M1[reg, q][ok] / w[ok, None]; sd = (d2[reg, q][ok] / w[ok, None] - mu ** 2).clamp_min(0).sqrt()
    key = (reg * P.NP + q)[ok]; uk, inv, cnt = torch.unique(key, return_inverse=True, return_counts=True); hk = h[ok]
    tm = torch.zeros(len(uk), 16, device=DEV, dtype=dt).index_add_(0, inv, hk) / cnt[:, None]
    tsd = (torch.zeros(len(uk), 16, device=DEV, dtype=dt).index_add_(0, inv, hk * hk) / cnt[:, None] - tm ** 2).clamp_min(0).sqrt()
    errm = (mu - tm[inv]).abs().mean(0) / hk.abs().mean(0).clamp_min(1e-9); sdr = sd.mean(0) / tsd[inv].mean(0).clamp_min(1e-9)
    print(f"step {t}: lists with a model node {ok.double().mean().item():.4f}")
    print("   mean abs error / mean abs value: " + " ".join(f"{a:.2f}" for a in errm.tolist()))
    print("   SD model / SD true:              " + " ".join(f"{a:.2f}" for a in sdr.tolist()), flush=True)

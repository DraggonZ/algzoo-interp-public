"""Development check of prop3 (sampled lists): per step, compare the model's per-node state mean and SD with the
sampled ones (same regime and grid cells of L and S; the model's look-ahead sub-regimes pooled), per neuron."""
import sys, json, numpy as np, torch
from algzoo.core import example
from prop3 import Prop, DEV, dt
G = int(sys.argv[1]); LA = json.loads(sys.argv[2]) if len(sys.argv) > 2 else []
W = [np.asarray(w, float) for w in example()]
P = Prop(W, G=G, lookahead=LA)
X = torch.randn(2_000_000, 10, device=DEV, dtype=dt, generator=torch.Generator(device=DEV).manual_seed(7))
cell = lambda v: torch.clamp((torch.special.ndtr(v) * G).long(), 0, G - 1)
wi, wh, wo = P.wi, P.wh, P.wo; n = len(X)
h = torch.relu(X[:, :1] * wi); L = X[:, 0].clone(); S = torch.full((n,), -1e9, device=DEV, dtype=dt)
pL = torch.zeros(n, dtype=torch.long, device=DEV); pS = torch.full((n,), -1, dtype=torch.long, device=DEV)
hh = torch.relu(P.v[:, None] * P.wi[None]); bits = P._bits(hh); st = {}
for b in torch.unique(bits).tolist():
    r = P._new(); m = (bits == b).to(dt)
    r[0][:, 0] = P.pw * m; r[1][:, 0] = P.pw * m[:, None] * hh; r[2][:, 0] = P.pw * m[:, None, None] * hh[:, :, None] * hh[:, None, :]
    st[(0, -1, b)] = r
for t in range(1, 10):
    x = X[:, t]; h = torch.relu(h @ wh.T + x[:, None] * wi)
    rec = x > L; sec = ~rec & (x > S)
    S = torch.where(rec, L, torch.where(sec, x, S)); pS = torch.where(rec, pL, torch.where(sec, t, pS))
    L = torch.where(rec, x, L); pL = torch.where(rec, t, pL)
    st = P.step(st, t)
    Wt = torch.zeros(10, 11, G, G + 1, device=DEV, dtype=dt); M1 = torch.zeros(10, 11, G, G + 1, 16, device=DEV, dtype=dt); M2 = M1.clone()
    for (a, b, _), o in st.items():
        Wt[a, b + 1] += o[0]; M1[a, b + 1] += o[1]; M2[a, b + 1] += torch.diagonal(o[2], dim1=-2, dim2=-1)
    iL = cell(L); iS = torch.where(S < -1e8, 0, cell(S) + 1)
    w = Wt[pL, pS + 1, iL, iS]; ok = w > 0
    mu = M1[pL, pS + 1, iL, iS][ok] / w[ok, None]; sd = (M2[pL, pS + 1, iL, iS][ok] / w[ok, None] - mu ** 2).clamp_min(0).sqrt()
    key = (((pL * 11 + pS + 1) * G + iL) * (G + 1) + iS)[ok]
    uk, inv, cnt = torch.unique(key, return_inverse=True, return_counts=True)
    hk = h[ok]
    tm = torch.zeros(len(uk), 16, device=DEV, dtype=dt).index_add_(0, inv, hk) / cnt[:, None]
    tsd = (torch.zeros(len(uk), 16, device=DEV, dtype=dt).index_add_(0, inv, hk * hk) / cnt[:, None] - tm ** 2).clamp_min(0).sqrt()
    # per list: model node mean vs sampled node mean
    errm = (mu - tm[inv]).abs().mean(0) / hk.abs().mean(0); sdr = sd.mean(0) / tsd[inv].mean(0)
    print(f"step {t}: share of lists with a model node {ok.double().mean().item():.4f}")
    print("   mean abs error / mean abs value: " + " ".join(f"{a:.2f}" for a in errm.tolist()))
    print("   SD model / SD true:              " + " ".join(f"{a:.2f}" for a in sdr.tolist()), flush=True)

"""Why is the uncertain zone as wide as it is? Continue training a copy of ARC (an analysis; nothing hand-built is
optimized) and watch the close calls. Settings: ARC's own recipe at its final rate (AdamW, lr 0.04/4 x 0.01 = 1e-4,
weight decay 0.001, batch 1024), and a cleaner gradient (batch 65,536, same rate); also 10x the rate. Checkpoints up to
10,000 steps. Tracked on 2 million fixed lists: accuracy, the third's win share by gap (the zone), the largest's wins,
the weight change and its share in the 20 most sensitive weight directions (restore.py's sensitivity matrix).
Predictions (written first): if the zone is unfinished training, near-tie third wins fall and accuracy rises within a
few thousand steps, with the change in the sensitive directions; if the dense solution is built in, little moves."""
import sys; sys.path.insert(0, "../sensitive"); sys.path.insert(0, "../../family")
import numpy as np, torch
from torch.func import vmap
from restore import w0, unflat, logits
from common import DEV, dt
g = torch.Generator(device=DEV).manual_seed(61); Xs = torch.randn(40000, 10, device=DEV, dtype=dt, generator=g); ys = Xs.argsort(1, descending=True)[:, 1]
M = torch.zeros(432, 432, device=DEV, dtype=dt)
for i in range(0, len(Xs), 2000):
    xb, yb = Xs[i:i + 2000], ys[i:i + 2000]; Jl = vmap(torch.func.jacrev(logits), in_dims=(None, 0))(w0, xb); S = vmap(logits, in_dims=(None, 0))(w0, xb)
    ar = torch.arange(len(xb), device=DEV); lead = S[ar, yb][:, None] - S; gl = Jl[ar, yb][:, None, :] - Jl
    m = (lead < 2) & (torch.arange(10, device=DEV)[None] != yb[:, None]); G = gl[m]; M += G.T @ G
ev, V = torch.linalg.eigh(M); V20 = V.flip(1)[:, :20]
ge = torch.Generator(device=DEV).manual_seed(7); XE = torch.randn(2_000_000, 10, device=DEV, dtype=dt, generator=ge)
oE = XE.argsort(1, descending=True); vE = XE.sort(1, descending=True).values; gap = (vE[:, 1] - vE[:, 2]) / vE[:, 1]; pos = vE[:, 1] > 0.1
def sc(v, X):
    wi, wh, wo = unflat(v); h = torch.zeros(len(X), 16, device=DEV, dtype=X.dtype)
    for t in range(10): h = torch.relu(h @ wh.T + X[:, t:t + 1] * wi)
    return h @ wo.T
def report(v, lab):
    a = sc(v, XE).argmax(1); d = v - w0
    z = [100 * (a[pos & (gap >= lo) & (gap < hi)] == oE[pos & (gap >= lo) & (gap < hi), 2]).double().mean().item() for lo, hi in ((0, 0.01), (0.01, 0.03), (0.03, 0.08), (0.08, 0.2))]
    share = ((V20.T @ d).pow(2).sum() / d.pow(2).sum().clamp(min=1e-30)).item() if d.abs().sum() > 0 else float("nan")
    print(f"{lab:34s} accuracy {100*(a == oE[:, 1]).double().mean().item():.3f}% | third wins at gaps <1/1-3/3-8/8-20%: {z[0]:.1f} {z[1]:.1f} {z[2]:.1f} {z[3]:.1f} | largest wins {100*(a == oE[:, 0]).double().mean().item():.2f}% | weight change RMS {d.pow(2).mean().sqrt().item():.5f}, in top-20 sensitive {share:.2f}", flush=True)
report(w0, "ARC")
for lab, batch, lr in (("ARC's recipe, final rate", 1024, 1e-4), ("big batch, final rate", 65536, 1e-4), ("big batch, 10x rate", 65536, 1e-3)):
    v = w0.clone().float().requires_grad_(True); opt = torch.optim.AdamW([v], lr=lr, weight_decay=0.001); gt = torch.Generator(device=DEV).manual_seed(99)
    for step in range(1, 10001):
        x = torch.randn(batch, 10, device=DEV, generator=gt); tgt = x.argsort(1)[:, -2]
        loss = torch.nn.functional.cross_entropy(sc(v, x), tgt); opt.zero_grad(); loss.backward(); opt.step()
        if step in (100, 1000, 10000): report(v.detach().double(), f"{lab}, {step} steps")

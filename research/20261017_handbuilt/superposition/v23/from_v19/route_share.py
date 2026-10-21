"""P10: share of each contender's lead through the keep gate (the second route), as steer/detector_share.py; and the
late-number bias at near ties, as steer/last_step.py (b = 0)."""
import sys; sys.path.insert(0, "/data/output/research/algzoo/20261017_handbuilt/family")
import torch
from common import DEV, dt, T, top3
from algzoo.handbuilt_v23_from_v19 import build_v23f

if __name__ == "__main__":
    g = torch.Generator(device=DEV).manual_seed(5); X = torch.randn(8_000_000, 10, device=DEV, dtype=dt, generator=g)
    v = X.sort(1, descending=True).values; l, s, t3 = top3(X)
    near = (v[:, 1] > 0) & ((v[:, 1] - v[:, 2]) < 0.05 * v[:, 1]); X, l, s, t3 = X[near], l[near], s[near], t3[near]
    W = T(build_v23f().params); KG = 2
    Xm = X[:50000].clone().requires_grad_(True); sm, tm = s[:50000], t3[:50000]; n = torch.arange(len(Xm), device=DEV)
    h = torch.zeros(len(Xm), 16, device=DEV, dtype=dt); Hs, Zs = [], []
    for t in range(10):
        z = h @ W[1].T + Xm[:, t:t + 1] * W[0]; Zs.append(z); h = torch.relu(z); Hs.append(h)
    sc = h @ W[2].T; D = sc[n, sm] - sc[n, tm]
    c, = torch.autograd.grad(D.sum(), Xm, retain_graph=True)
    GZ = torch.autograd.grad(D.sum(), Zs, retain_graph=True)
    via = torch.zeros_like(Xm)
    for t in range(10):
        gh = (GZ[t + 1] @ W[1])[:, KG] if t < 9 else (W[2][sm, KG] - W[2][tm, KG])
        J, = torch.autograd.grad(Hs[t][:, KG].sum(), Xm, retain_graph=True)
        via += gh[:, None] * J * Xm.detach()
    cx = c * Xm.detach(); fs = via[n, sm] / cx[n, sm]; ft = via[n, tm] / cx[n, tm]
    print(f"share through the keep gate, median: second {fs.median():+.2f}, third {ft.median():+.2f}")
    for p in range(10):
        print(f"   contender at step {p}: second {fs[sm == p].median():+.2f}, third {ft[tm == p].median():+.2f}")
    p = (torch.relu if False else None)
    hh = torch.zeros(len(X), 16, device=DEV, dtype=dt)
    for t in range(10): hh = torch.relu(hh @ W[1].T + X[:, t:t + 1] * W[0])
    a = (hh @ W[2].T).argmax(1); late = torch.where(s > t3, s, t3); m9 = late == 9
    print(f"near ties with a contender at step 9: it wins {100*(a[m9] == 9).double().mean().item():.1f}% (truly second {100*(s[m9] == 9).double().mean().item():.1f}%), right {100*(a[m9] == s[m9]).double().mean().item():.1f}%; all near ties right {100*(a == s).double().mean().item():.2f}%")

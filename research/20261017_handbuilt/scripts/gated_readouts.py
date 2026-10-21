"""Is the nonlinear part of reading ARC's memory 'which context am I in'? Fit separate linear readouts per context,
with the context given by an oracle: (a) where the largest is (10 readouts); (b) whether the second came before or
after the largest (2); (c) both (20); (d) where the largest and the second are (90). Each starts from ARC's readout and
is fitted by cross-entropy on 4M lists; errors on 2M fresh lists. Also: can the largest's position be read linearly
from the final state (so a two-stage linear reader could do this without an oracle)?"""
import torch as th

from algzoo.core import example

dev = "cuda"; th.manual_seed(0)
wi, wh, wo = (th.tensor(w, device=dev, dtype=th.float32) for w in example())


def final_h(X):
    h = th.zeros(len(X), 16, device=dev)
    for t in range(10):
        h = th.relu(h @ wh.T + X[:, t:t + 1] * wi)
    return h


g = th.Generator(device=dev).manual_seed(1)
Xtr = th.randn(4_000_000, 10, device=dev, generator=g); Xte = th.randn(2_000_000, 10, device=dev, generator=g)
Htr, Hte = final_h(Xtr), final_h(Xte)
otr, ote = Xtr.argsort(1, descending=True), Xte.argsort(1, descending=True)
scale = Htr.std(0) + 1e-6


def ctx(o, kind):
    p1, p2 = o[:, 0], o[:, 1]
    return {"largest position": p1, "order": (p2 > p1).long(), "largest position and order": p1 * 2 + (p2 > p1).long(),
            "largest and second positions": p1 * 10 + p2}[kind]


def fit_gated(kind, epochs=8):
    ctr, cte = ctx(otr, kind), ctx(ote, kind); K = int(max(ctr.max(), cte.max()).item()) + 1
    Wg = (wo * scale).unsqueeze(0).repeat(K, 1, 1).clone().requires_grad_(True)
    opt = th.optim.Adam([Wg], lr=3e-3)
    for ep in range(epochs):
        perm = th.randperm(len(Xtr), device=dev)
        for i in range(0, len(Xtr), 16384):
            j = perm[i:i + 16384]
            logits = th.einsum("nd,nkd->nk", Htr[j] / scale, Wg[ctr[j]])
            loss = th.nn.functional.cross_entropy(logits, otr[j, 1]); opt.zero_grad(); loss.backward(); opt.step()
        for p in opt.param_groups:
            p["lr"] *= 0.7
    with th.no_grad():
        ans = th.einsum("nd,nkd->nk", Hte / scale, Wg[cte]).argmax(1)
    return 100 * (ans != ote[:, 1]).float().mean().item()


print(f"ARC's readout: {100*((Hte @ wo.T).argmax(1) != ote[:, 1]).float().mean().item():.2f}%   (best single linear 4.41%, three-layer reader 2.35%)")
for kind in ("order", "largest position", "largest position and order", "largest and second positions"):
    print(f"separate linear readouts by {kind} (oracle): {fit_gated(kind):.2f}%")
lin = th.nn.Linear(16, 10).to(dev); opt = th.optim.Adam(lin.parameters(), lr=3e-3)
for ep in range(6):
    perm = th.randperm(len(Xtr), device=dev)
    for i in range(0, len(Xtr), 16384):
        j = perm[i:i + 16384]
        loss = th.nn.functional.cross_entropy(lin(Htr[j] / scale), otr[j, 0]); opt.zero_grad(); loss.backward(); opt.step()
with th.no_grad():
    acc = (lin(Hte / scale).argmax(1) == ote[:, 0]).float().mean().item()
print(f"largest's position read linearly from the final state: {100*acc:.1f}% correct")

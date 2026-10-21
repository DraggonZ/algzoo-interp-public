"""Is the nonlinear part of reading 'where the largest is'? Separate linear readouts per context, with contexts that
do not reveal which of the second and third wins: (a) the largest's position (10); (b) the class both-before / both-after
/ mixed (3); (c) both (30). Oracle contexts; then (d) the same with the largest's position read from the final state by
a nonlinear reader (no oracle). Errors on 2M fresh lists (best single linear 4.38%, nonlinear reader 2.35%)."""
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
Htr, Hte = final_h(Xtr), final_h(Xte); scale = Htr.std(0) + 1e-6
otr, ote = Xtr.argsort(1, descending=True), Xte.argsort(1, descending=True)


def cls(o):
    b2, b3 = o[:, 1] < o[:, 0], o[:, 2] < o[:, 0]
    return th.where(b2 & b3, 0, th.where(~b2 & ~b3, 1, 2))


def fit_gated(ctr, K, epochs=8):
    Wg = (wo * scale).unsqueeze(0).repeat(K, 1, 1).clone().requires_grad_(True); opt = th.optim.Adam([Wg], lr=3e-3)
    for ep in range(epochs):
        perm = th.randperm(len(Xtr), device=dev)
        for i in range(0, len(Xtr), 16384):
            j = perm[i:i + 16384]
            loss = th.nn.functional.cross_entropy(th.einsum("nd,nkd->nk", Htr[j] / scale, Wg[ctr[j]]), otr[j, 1])
            opt.zero_grad(); loss.backward(); opt.step()
        for p in opt.param_groups:
            p["lr"] *= 0.7
    return Wg.detach()


def err(Wg, cte):
    with th.no_grad():
        return 100 * (th.einsum("nd,nkd->nk", Hte / scale, Wg[cte]).argmax(1) != ote[:, 1]).float().mean().item()


ctx = {"largest position": (otr[:, 0], ote[:, 0], 10), "class": (cls(otr), cls(ote), 3),
       "largest position x class": (otr[:, 0] * 3 + cls(otr), ote[:, 0] * 3 + cls(ote), 30)}
W = {}
for k, (ctr, cte, K) in ctx.items():
    W[k] = fit_gated(ctr, K); print(f"(oracle) separate readouts by {k}: {err(W[k], cte):.2f}%", flush=True)
# (d) largest position read by a small nonlinear reader, class kept as oracle and then also read
def reader(y, K):
    net = th.nn.Sequential(th.nn.Linear(16, 256), th.nn.ReLU(), th.nn.Linear(256, 256), th.nn.ReLU(), th.nn.Linear(256, K)).to(dev)
    opt = th.optim.Adam(net.parameters(), lr=3e-3)
    for ep in range(8):
        perm = th.randperm(len(Htr), device=dev)
        for i in range(0, len(Htr), 16384):
            j = perm[i:i + 16384]
            loss = th.nn.functional.cross_entropy(net(Htr[j] / scale), y[j]); opt.zero_grad(); loss.backward(); opt.step()
        for p in opt.param_groups:
            p["lr"] *= 0.75
    with th.no_grad():
        return net(Hte / scale).argmax(1)
p1_hat = reader(otr[:, 0], 10); c_hat = reader(cls(otr), 3)
print(f"largest position read from the state: {100*(p1_hat == ote[:, 0]).float().mean().item():.1f}% correct; class read: {100*(c_hat == cls(ote)).float().mean().item():.1f}% correct")
print(f"(no oracle) readouts by read largest position: {err(W['largest position'], p1_hat):.2f}%; by read position x read class: {err(W['largest position x class'], p1_hat * 3 + c_hat):.2f}%")

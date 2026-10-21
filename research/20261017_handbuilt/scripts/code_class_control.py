"""Control for 'two codes, one linear compromise'. The oracle bit 'second before or after the
largest' leaks the answer on mixed lists (second and third on opposite sides of the largest). Gate the readouts instead
on a context that treats second and third alike: both before the largest, both after, or mixed (3 readouts, oracle).
If the gain survives on same-side lists, the two codes are real; if it sits in mixed lists, it was leakage.
Also split the order-gated readouts' errors by these classes. Lists with at least three positive numbers decide the
class; others are their own class."""
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


def classes(o):
    b2, b3 = o[:, 1] < o[:, 0], o[:, 2] < o[:, 0]
    return th.where(b2 & b3, 0, th.where(~b2 & ~b3, 1, 2))           # 0 both before, 1 both after, 2 mixed


def order(o):
    return (o[:, 1] < o[:, 0]).long()


def fit_gated(ctr, K, epochs=8):
    Wg = (wo * scale).unsqueeze(0).repeat(K, 1, 1).clone().requires_grad_(True)
    opt = th.optim.Adam([Wg], lr=3e-3)
    for ep in range(epochs):
        perm = th.randperm(len(Xtr), device=dev)
        for i in range(0, len(Xtr), 16384):
            j = perm[i:i + 16384]
            loss = th.nn.functional.cross_entropy(th.einsum("nd,nkd->nk", Htr[j] / scale, Wg[ctr[j]]), otr[j, 1])
            opt.zero_grad(); loss.backward(); opt.step()
        for p in opt.param_groups:
            p["lr"] *= 0.7
    return Wg.detach()


cte = classes(ote); names = ["both before", "both after", "mixed"]
wrong = {}
W1 = fit_gated(th.zeros(len(Xtr), dtype=th.long, device=dev), 1)[0]
Wo = fit_gated(order(otr), 2); Wc = fit_gated(classes(otr), 3)
with th.no_grad():
    wrong["ARC's readout"] = (Hte @ wo.T).argmax(1) != ote[:, 1]
    wrong["single best linear"] = (Hte / scale @ W1.T).argmax(1) != ote[:, 1]
with th.no_grad():
    wrong["order-gated (leaky oracle)"] = th.einsum("nd,nkd->nk", Hte / scale, Wo[order(ote)]).argmax(1) != ote[:, 1]
    wrong["class-gated (control oracle)"] = th.einsum("nd,nkd->nk", Hte / scale, Wc[cte]).argmax(1) != ote[:, 1]
print("error, overall and by class (both before / both after / mixed; shares of lists " +
      " / ".join(f"{100*(cte == k).float().mean().item():.0f}%" for k in range(3)) + ")")
for k, w in wrong.items():
    print(f"{k:30s} {100*w.float().mean().item():5.2f}%  | " + " / ".join(f"{100*w[cte == c].float().mean().item():5.2f}%" for c in range(3)))

"""What is in the last 1.2 points? Lists the nonlinear reader (reader_three-layer.pt) gets right but the two-stage
reader (largest position read nonlinearly, then that position's linear readout) gets wrong, against all lists.
Features: second-third gap, class, a fourth number above 0.75 of the second (could be kept), the last input among the
top three, number of positive inputs, mistake kind of the two-stage reader."""
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
ck = th.load("reader_three-layer.pt"); scale = ck["scale"]
otr, ote = Xtr.argsort(1, descending=True), Xte.argsort(1, descending=True)
Wg = (wo * scale).unsqueeze(0).repeat(10, 1, 1).clone().requires_grad_(True); opt = th.optim.Adam([Wg], lr=3e-3)
for ep in range(8):
    perm = th.randperm(len(Xtr), device=dev)
    for i in range(0, len(Xtr), 16384):
        j = perm[i:i + 16384]
        loss = th.nn.functional.cross_entropy(th.einsum("nd,nkd->nk", Htr[j] / scale, Wg[otr[j, 0]]), otr[j, 1]); opt.zero_grad(); loss.backward(); opt.step()
    for p in opt.param_groups:
        p["lr"] *= 0.7
net = th.nn.Sequential(th.nn.Linear(16, 256), th.nn.ReLU(), th.nn.Linear(256, 256), th.nn.ReLU(), th.nn.Linear(256, 10)).to(dev)
o2 = th.optim.Adam(net.parameters(), lr=3e-3)
for ep in range(8):
    perm = th.randperm(len(Htr), device=dev)
    for i in range(0, len(Htr), 16384):
        j = perm[i:i + 16384]
        loss = th.nn.functional.cross_entropy(net(Htr[j] / scale), otr[j, 0]); o2.zero_grad(); loss.backward(); o2.step()
    for p in o2.param_groups:
        p["lr"] *= 0.75
big = th.nn.Sequential(th.nn.Linear(16, 512), th.nn.ReLU(), th.nn.Linear(512, 512), th.nn.ReLU(), th.nn.Linear(512, 512), th.nn.ReLU(), th.nn.Linear(512, 10)).to(dev)
big.load_state_dict(ck["state"])
with th.no_grad():
    p1 = net(Hte / scale).argmax(1)
    a2 = th.einsum("nd,nkd->nk", Hte / scale, Wg.detach()[p1]).argmax(1)
    aN = big(Hte / scale).argmax(1)
y = ote[:, 1]; v = Xte.gather(1, ote[:, :4])
fixed = (a2 != y) & (aN == y)
gap = th.where(v[:, 1] > 0, (v[:, 1] - v[:, 2]) / v[:, 1].abs(), th.full_like(v[:, 0], 9.0))
gap12 = th.where(v[:, 0] > 0, (v[:, 0] - v[:, 1]) / v[:, 0].abs(), th.full_like(v[:, 0], 9.0))
b2, b3 = ote[:, 1] < ote[:, 0], ote[:, 2] < ote[:, 0]
feats = {"second-third gap < 3%": gap < 0.03, "second-third gap 3-10%": (gap >= 0.03) & (gap < 0.1), "largest-second gap < 10%": gap12 < 0.1,
         "both before largest": b2 & b3, "both after largest": ~b2 & ~b3, "mixed": b2 != b3,
         "fourth >= 0.75 of second": (v[:, 3] >= 0.75 * v[:, 1]) & (v[:, 1] > 0), "last input in top three": (ote[:, :3] == 9).any(1),
         "at most one positive": (Xte > 0).sum(1) <= 1, "two-stage answers the third": a2 == ote[:, 2], "two-stage answers the largest": a2 == ote[:, 0]}
print(f"two-stage error {100*(a2 != y).float().mean().item():.2f}%, nonlinear {100*(aN != y).float().mean().item():.2f}%; "
      f"fixed by the nonlinear reader {100*fixed.float().mean().item():.2f} pts, newly broken {100*((a2 == y) & (aN != y)).float().mean().item():.2f} pts")
print("feature | share among fixed lists | share among all lists | lift")
for k, m in feats.items():
    a, b = m[fixed].float().mean().item(), m.float().mean().item()
    print(f"{k:32s} | {100*a:5.1f}% | {100*b:5.1f}% | {a/max(b,1e-9):5.1f}")

# Kept votes: count numbers kept under the account's rule (at arrival, at least 0.75 of the second-largest so far,
# or a new largest), excluding the largest itself.
X = Xte; n = len(X); m1 = th.full((n,), -1e9, device=dev); m2 = th.full((n,), -1e9, device=dev); kept = th.zeros(n, device=dev)
for t in range(10):
    x = X[:, t]
    is_new_max = x > m1
    k = (x > 0) & (is_new_max | (x >= 0.75 * th.clamp(m2, min=0)))
    kept += k.float()
    m2 = th.where(is_new_max, m1, th.maximum(m2, x)); m1 = th.maximum(m1, x)
kept = kept - 1                                              # the largest itself is always kept if positive
with th.no_grad():
    aA = (Hte @ wo.T).argmax(1)
print("\nkept numbers besides the largest | share of lists | ARC's readout | two-stage | nonlinear reader")
for c in range(0, 6):
    m = (kept == c) if c < 5 else (kept >= 5)
    if m.sum() == 0:
        continue
    e = lambda a: 100 * (a[m] != y[m]).float().mean().item()
    print(f"{c if c < 5 else '5+'} | {100*m.float().mean().item():5.1f}% | {e(aA):5.2f}% | {e(a2):5.2f}% | {e(aN):5.2f}%")

"""Two codes, one reader. (1) Can 'second before or after the largest' be read linearly from ARC's final state?
(2) A two-stage reader: read the order linearly, then apply the readout fitted for that order. (3) How badly does each
order's readout do on the other order's lists, and ARC's single readout on each? Errors on 2M fresh lists."""
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
btr, bte = (otr[:, 1] < otr[:, 0]).long(), (ote[:, 1] < ote[:, 0]).long()    # 1 = second before largest


def fit(H, y, W0, epochs=8, lr=3e-3):
    W = W0.clone().requires_grad_(True); opt = th.optim.Adam([W], lr=lr)
    for ep in range(epochs):
        perm = th.randperm(len(H), device=dev)
        for i in range(0, len(H), 16384):
            j = perm[i:i + 16384]
            loss = th.nn.functional.cross_entropy(H[j] @ W.T, y[j]); opt.zero_grad(); loss.backward(); opt.step()
        for p in opt.param_groups:
            p["lr"] *= 0.7
    return W.detach()


Hs_tr = th.cat([Htr / scale, th.ones(len(Htr), 1, device=dev)], 1); Hs_te = th.cat([Hte / scale, th.ones(len(Hte), 1, device=dev)], 1)
Wb = fit(Hs_tr, btr, th.zeros(2, 17, device=dev))
pred_b = (Hs_te @ Wb.T).argmax(1)
print(f"order (second before / after largest) read linearly: {100*(pred_b == bte).float().mean().item():.1f}% correct")
W0 = wo * scale
R = {k: fit(Htr[btr == k] / scale, otr[btr == k, 1], W0) for k in (0, 1)}
th.save({"after": (R[0] / scale).cpu(), "before": (R[1] / scale).cpu()}, "two_code_readouts.pt")
err = lambda W, m: 100 * (((Hte[m] / scale) @ W.T).argmax(1) != ote[m, 1]).float().mean().item()
for k, lab in ((1, "second before largest"), (0, "second after largest")):
    m = bte == k
    print(f"{lab} ({100*m.float().mean().item():.0f}% of lists): ARC's readout {err(W0, m):.2f}%, its own readout {err(R[k], m):.2f}%, the other order's readout {err(R[1-k], m):.2f}%")
ans = th.where(pred_b == 1, ((Hte / scale) @ R[1].T).argmax(1), ((Hte / scale) @ R[0].T).argmax(1))
print(f"two-stage linear reader (order read from the state, then that order's readout): {100*(ans != ote[:, 1]).float().mean().item():.2f}% error (ARC 4.68%)")

# (4) Nonlinear read of the order, and the two-stage reader with it.
mlp = th.nn.Sequential(th.nn.Linear(16, 256), th.nn.ReLU(), th.nn.Linear(256, 256), th.nn.ReLU(), th.nn.Linear(256, 2)).to(dev)
opt = th.optim.Adam(mlp.parameters(), lr=3e-3)
for ep in range(10):
    perm = th.randperm(len(Htr), device=dev)
    for i in range(0, len(Htr), 16384):
        j = perm[i:i + 16384]
        loss = th.nn.functional.cross_entropy(mlp(Htr[j] / scale), btr[j]); opt.zero_grad(); loss.backward(); opt.step()
    for p in opt.param_groups:
        p["lr"] *= 0.75
with th.no_grad():
    pb = mlp(Hte / scale).argmax(1)
print(f"order read by a small nonlinear reader: {100*(pb == bte).float().mean().item():.1f}% correct")
ans = th.where(pb == 1, ((Hte / scale) @ R[1].T).argmax(1), ((Hte / scale) @ R[0].T).argmax(1))
print(f"two-stage reader (nonlinear order, then linear readout): {100*(ans != ote[:, 1]).float().mean().item():.2f}% error")
wrong_order = pb != bte
print(f"  order misread in {100*wrong_order.float().mean().item():.1f}% of lists; answer error there {100*(ans != ote[:, 1])[wrong_order].float().mean().item():.1f}%")
# (5) Where the two codes differ: per neuron, size of the difference between the two readouts (in units of activity spread).
D = (R[1] - R[0]).abs().sum(0); A = (R[1].abs() + R[0].abs()).sum(0) / 2
print("neuron: readout difference / mean readout size  ->", "  ".join(f"{k}:{(D[k]/A[k]).item():.2f}" for k in range(16)))

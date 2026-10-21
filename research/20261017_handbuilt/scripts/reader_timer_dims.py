"""How many dimensions of the twelve timer neurons does a nonlinear reader need? Features: the top k principal
directions of the twelve neurons' final activity (on random lists) plus neurons 1, 6, 7, 8 and 5. Same reader as
reader_subsets.py."""
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
ytr, yte = Xtr.argsort(1, descending=True)[:, 1], Xte.argsort(1, descending=True)[:, 1]
T = [0, 2, 3, 4, 9, 10, 11, 12, 13, 14, 15]; KEEP = [1, 6, 7, 8, 5]        # timer neurons other than 5; kept as is
mu = Htr[:, T].mean(0); U, S, V = th.linalg.svd(Htr[:200_000, T] - mu, full_matrices=False)
print("share of timer-neuron variance in top directions:", " ".join(f"{(S[:k]**2).sum().item()/(S**2).sum().item():.3f}" for k in range(1, 12)))
for k in (2, 3, 4, 6, 8, 11):
    feat = lambda H: th.cat([(H[:, T] - mu) @ V[:k].T, H[:, KEEP]], 1)
    Ftr, Fte = feat(Htr), feat(Hte); sc = Ftr.std(0) + 1e-6
    net = th.nn.Sequential(th.nn.Linear(Ftr.shape[1], 512), th.nn.ReLU(), th.nn.Linear(512, 512), th.nn.ReLU(), th.nn.Linear(512, 512), th.nn.ReLU(), th.nn.Linear(512, 10)).to(dev)
    opt = th.optim.Adam(net.parameters(), lr=2e-3)
    for ep in range(20):
        perm = th.randperm(len(Ftr), device=dev)
        for i in range(0, len(Ftr), 8192):
            j = perm[i:i + 8192]
            loss = th.nn.functional.cross_entropy(net(Ftr[j] / sc), ytr[j]); opt.zero_grad(); loss.backward(); opt.step()
        for p in opt.param_groups:
            p["lr"] *= 0.85
    with th.no_grad():
        print(f"top {k:2d} timer directions + 1, 6, 7, 8, 5: {100*(net(Fte / sc).argmax(1) != yte).float().mean().item():.2f}% error")

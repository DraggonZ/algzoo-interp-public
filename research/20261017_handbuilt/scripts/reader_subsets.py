"""Which neurons does a nonlinear reader of ARC's final state need? Train the same three-layer reader (512 wide) on
subsets of the 16 final activities and report error on 2M fresh lists. Also ARC's readout restricted to the subset
is not meaningful, so only the nonlinear reader is compared with the full-state reader (about 2.35%)."""
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
ytr, yte = Xtr.argsort(1, descending=True)[:, 1], Xte.argsort(1, descending=True)[:, 1]
LOW = [1, 6, 7, 8]; DIAL = [0, 2, 3, 4, 5, 9, 10, 11, 12, 13, 14, 15]
subsets = {"all 16": list(range(16)), "without 1, 6, 7, 8": DIAL, "only 1, 6, 7, 8": LOW,
           "without 2 and 4 (largest so far)": [k for k in range(16) if k not in (2, 4)],
           "without 5 (gate)": [k for k in range(16) if k != 5], "without 3": [k for k in range(16) if k != 3],
           "without 9": [k for k in range(16) if k != 9],
           "dial plane (2 numbers) + 1, 6, 7, 8 + 4, 5, 9": None}
axes = None
for name, S in subsets.items():
    if S is None:
        import json
        m = json.load(open("viz/model.json")); A = th.tensor(m["dial_axes"], device=dev, dtype=th.float32)
        mu = th.tensor(m["dial_mean"], device=dev, dtype=th.float32); idx = m["dial_neurons"]
        feat = lambda H: th.cat([(H[:, idx] - mu) @ A.T, H[:, LOW + [4, 5, 9]]], 1)
    else:
        feat = (lambda S: (lambda H: H[:, S]))(S)
    Ftr, Fte = feat(Htr), feat(Hte); sc = Ftr.std(0) + 1e-6
    d = Ftr.shape[1]
    net = th.nn.Sequential(th.nn.Linear(d, 512), th.nn.ReLU(), th.nn.Linear(512, 512), th.nn.ReLU(), th.nn.Linear(512, 512), th.nn.ReLU(), th.nn.Linear(512, 10)).to(dev)
    opt = th.optim.Adam(net.parameters(), lr=2e-3)
    for ep in range(20):
        perm = th.randperm(len(Ftr), device=dev)
        for i in range(0, len(Ftr), 8192):
            j = perm[i:i + 8192]
            loss = th.nn.functional.cross_entropy(net(Ftr[j] / sc), ytr[j]); opt.zero_grad(); loss.backward(); opt.step()
        for p in opt.param_groups:
            p["lr"] *= 0.85
    with th.no_grad():
        err = 100 * (net(Fte / sc).argmax(1) != yte).float().mean().item()
    print(f"{name:48s} ({d:2d} inputs): {err:.2f}% error")

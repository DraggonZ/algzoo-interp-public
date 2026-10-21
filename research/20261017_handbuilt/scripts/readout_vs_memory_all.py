"""Readout versus memory across the 22 trained networks: own readout, best linear readout (from the own), and a
nonlinear reader (three layers of 512) on the final state; 4M training lists, 2M fresh test lists."""
from pathlib import Path

import torch as th

from algzoo.core import example, load, zoo

dev = "cuda"
models = {"ARC": example(), **{f"zoo{s}": zoo(16, 10, s) for s in range(5)}}
for f in sorted(Path("/data/output/research/algzoo/20261016_run3/models").glob("2nd_argmax_16_10_*_1*.pth")):
    models["s" + f.stem.split("_")[-1]] = load(str(f))
g = th.Generator(device=dev).manual_seed(1)
Xtr = th.randn(4_000_000, 10, device=dev, generator=g); Xte = th.randn(2_000_000, 10, device=dev, generator=g)
ytr, yte = Xtr.argsort(1, descending=True)[:, 1], Xte.argsort(1, descending=True)[:, 1]


def train(model, Ftr, lr, epochs, decay):
    opt = th.optim.Adam(model.parameters(), lr=lr)
    for ep in range(epochs):
        perm = th.randperm(len(Ftr), device=dev)
        for i in range(0, len(Ftr), 8192):
            j = perm[i:i + 8192]
            loss = th.nn.functional.cross_entropy(model(Ftr[j]), ytr[j]); opt.zero_grad(); loss.backward(); opt.step()
        for p in opt.param_groups:
            p["lr"] *= decay
    return model


print("network | own readout | best linear | nonlinear reader | share of own errors recoverable by reading")
for name, P in models.items():
    th.manual_seed(0)
    wi, wh, wo = (th.tensor(w, device=dev, dtype=th.float32) for w in P)
    def fh(X):
        h = th.zeros(len(X), 16, device=dev)
        for t in range(10):
            h = th.relu(h @ wh.T + X[:, t:t + 1] * wi)
        return h
    Htr, Hte = fh(Xtr), fh(Xte); scale = Htr.std(0) + 1e-6
    own = 100 * ((Hte @ wo.T).argmax(1) != yte).float().mean().item()
    lin = th.nn.Linear(16, 10, bias=False).to(dev)
    with th.no_grad():
        lin.weight.copy_(wo * scale)
    lin = train(lin, Htr / scale, 3e-4, 8, 0.85)
    net = th.nn.Sequential(th.nn.Linear(16, 512), th.nn.ReLU(), th.nn.Linear(512, 512), th.nn.ReLU(), th.nn.Linear(512, 512), th.nn.ReLU(), th.nn.Linear(512, 10)).to(dev)
    net = train(net, Htr / scale, 2e-3, 20, 0.85)
    with th.no_grad():
        el = 100 * (lin(Hte / scale).argmax(1) != yte).float().mean().item()
        en = 100 * (net(Hte / scale).argmax(1) != yte).float().mean().item()
    print(f"{name:5s} | {own:5.2f}% | {el:5.2f}% | {en:5.2f}% | {100*(own-en)/own:3.0f}%", flush=True)

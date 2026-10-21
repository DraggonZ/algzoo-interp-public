"""v2: the nonlinear reader also gets the network's own (designed) scores, so it can at least copy the designed
readout; any improvement beyond that is a genuine reading gap. Baseline for 'much of the error is in reading': the same readers on the hand-built network's final state
(build_v2 at gains 100 and 1000, readout designed by hand). If a nonlinear reader also removes a large share of its
mistakes, the finding is generic to linear readouts of a 16-neuron ReLU state; if not, it is a property of the trained
networks."""
import numpy as np
import torch as th

from algzoo import handbuilt as hb

dev = "cuda"
g = th.Generator(device=dev).manual_seed(1)
Xtr = th.randn(4_000_000, 10, device=dev, generator=g); Xte = th.randn(2_000_000, 10, device=dev, generator=g)
ytr, yte = Xtr.argsort(1, descending=True)[:, 1], Xte.argsort(1, descending=True)[:, 1]
for gain in (30.0, 100.0, 1000.0):
    th.manual_seed(0)
    net = hb.build_v2(gain, gain); wi, wh, wo = (th.tensor(np.asarray(w), device=dev, dtype=th.float64) for w in net.params)
    def fh(X):
        h = th.zeros(len(X), len(wi), device=dev, dtype=th.float64)
        for t in range(10):
            h = th.relu(h @ wh.T + X[:, t:t + 1].double() * wi)
        return h
    Htr, Hte = fh(Xtr), fh(Xte)
    own = 100 * ((Hte @ wo.T).argmax(1) != yte).double().mean().item()
    Htr, Hte = Htr.float(), Hte.float(); scale = Htr.std(0) + 1e-6
    lin = th.nn.Linear(len(wi), 10, bias=False).to(dev)
    with th.no_grad():
        lin.weight.copy_(wo.float() * scale)                 # start from the hand-designed readout
    med = Htr.clamp(min=0).median(0).values + 1e-3 * scale
    wof = wo.float(); ssc = (Htr @ wof.T).std() + 1e-6
    feat = lambda H: th.cat([(H @ wof.T) / ssc, H / scale, th.log1p(H / med)], 1)   # own scores + activity + log activity
    r = th.nn.Sequential(th.nn.Linear(10 + 2 * len(wi), 512), th.nn.ReLU(), th.nn.Linear(512, 512), th.nn.ReLU(), th.nn.Linear(512, 512), th.nn.ReLU(), th.nn.Linear(512, 10)).to(dev)
    out = []
    for model, lr, ep, F in ((lin, 3e-4, 8, lambda H: H / scale), (r, 2e-3, 20, feat)):
        opt = th.optim.Adam(model.parameters(), lr=lr)
        for e in range(ep):
            perm = th.randperm(len(Htr), device=dev)
            for i in range(0, len(Htr), 8192):
                j = perm[i:i + 8192]
                loss = th.nn.functional.cross_entropy(model(F(Htr[j])), ytr[j]); opt.zero_grad(); loss.backward(); opt.step()
            for p in opt.param_groups:
                p["lr"] *= 0.85
        with th.no_grad():
            out.append(100 * (model(F(Hte)).argmax(1) != yte).float().mean().item())
    print(f"hand-built, gains {gain:g}: own readout {own:.2f}%, fitted linear {out[0]:.2f}%, nonlinear reader {out[1]:.2f}%; share of own errors recoverable {100*(own-out[1])/own:.0f}%", flush=True)

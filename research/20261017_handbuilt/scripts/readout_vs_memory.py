"""Where do ARC's mistakes come from: the memory (the final hidden state) or the reading (the output weights)?
Keep ARC's input and recurrent weights; fit the best output weights on the final hidden state of 4 million random
lists (multinomial logistic regression, linear like the real readout), and also a small two-layer reader (upper
bound on what the memory holds). Evaluate on 2 million fresh lists, with mistakes by kind
(A at most one positive, B third, C largest, D other)."""
import numpy as np
import torch as th

from algzoo.core import example

dev = "cuda"; th.manual_seed(0)
wi, wh, wo = (th.tensor(w, device=dev, dtype=th.float32) for w in example())


def final_h(X):
    h = th.zeros(len(X), 16, device=dev)
    for t in range(10):
        h = th.relu(h @ wh.T + X[:, t:t + 1] * wi)
    return h


def kinds(X, ans):
    o = X.argsort(1, descending=True); wrong = ans != o[:, 1]; few = (X > 0).sum(1) <= 1
    k = lambda t: 100 * t.float().mean().item()
    A = k(wrong & few); B = k(wrong & ~few & (ans == o[:, 2])); C = k(wrong & ~few & (ans == o[:, 0]))
    return f"error {k(wrong):.2f} (A {A:.2f}, B {B:.2f}, C {C:.2f}, other {k(wrong) - A - B - C:.2f})"


g = th.Generator(device=dev).manual_seed(1)
Xtr = th.randn(4_000_000, 10, device=dev, generator=g); Xte = th.randn(2_000_000, 10, device=dev, generator=g)
Htr, Hte = final_h(Xtr), final_h(Xte); ytr = Xtr.argsort(1, descending=True)[:, 1]
print("ARC's own readout:                 ", kinds(Xte, (Hte @ wo.T).argmax(1)))
scale = Htr.std(0) + 1e-6
lin = th.nn.Linear(16, 10, bias=False)
with th.no_grad():
    lin.weight.copy_(wo * scale)                      # start from ARC's own readout (inputs are scaled)
big = lambda w: th.nn.Sequential(th.nn.Linear(16, w), th.nn.ReLU(), th.nn.Linear(w, w), th.nn.ReLU(), th.nn.Linear(w, w), th.nn.ReLU(), th.nn.Linear(w, 10))
for name, model, lr, epochs in (("best linear readout (from ARC's):", lin, 3e-4, 12),
                                ("two-layer reader (256):           ", th.nn.Sequential(th.nn.Linear(16, 256), th.nn.ReLU(), th.nn.Linear(256, 256), th.nn.ReLU(), th.nn.Linear(256, 10)), 3e-3, 12),
                                ("three-layer reader (512), longer:  ", big(512), 2e-3, 30)):
    model = model.to(dev); opt = th.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        perm = th.randperm(len(Xtr), device=dev)
        for i in range(0, len(Xtr), 8192):
            j = perm[i:i + 8192]
            loss = th.nn.functional.cross_entropy(model(Htr[j] / scale), ytr[j])
            opt.zero_grad(); loss.backward(); opt.step()
        for p in opt.param_groups:
            p["lr"] *= 0.85
    with th.no_grad():
        print(name, kinds(Xte, model(Hte / scale).argmax(1)))
    th.save({"state": model.state_dict(), "scale": scale}, f"reader_{name.split()[0]}.pt")

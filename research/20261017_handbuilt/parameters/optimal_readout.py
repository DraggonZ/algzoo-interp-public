"""Is ARC's readout the cross-entropy optimum for ARC's own final hidden states?

Fit set: 2^20 fresh lists (seed 11), ARC's final states h (float64). Minimise mean cross-entropy over the 10x16 readout
by L-BFGS, from ARC's readout and from zero. The loss is invariant to adding one vector to all rows, so matrices are
compared after centring over positions. Accuracy cost on the usual separate 2^22 lists (seed 7), paired with ARC."""
import numpy as np
import torch as th
from algzoo.core import example, forward

dev = "cuda"
wi, wh, wo = (np.asarray(p, float) for p in example())
N = 2**22
NF = 2**20  # fit set size (160 numbers to fit)


def states(seed, n=N):
    x = th.randn(n, 10, generator=th.Generator(device=dev).manual_seed(seed), device=dev, dtype=th.float64)
    _, H = forward((wi, wh, wo), x, return_hidden=True)
    return H[:, -1].contiguous(), x.argsort(-1)[:, -2]


hf, yf = states(11, NF)
ht, yt = states(7)
acc = lambda W: ((ht @ th.as_tensor(W, device=dev).T).argmax(-1) == yt).double().mean().item()
ce = lambda W, h, y: th.nn.functional.cross_entropy(h @ th.as_tensor(W, device=dev).T, y).item()
acc0 = acc(wo)
cen = lambda W: W - W.mean(0)


def fit(W_init):
    W = th.tensor(W_init, device=dev, requires_grad=True)
    opt = th.optim.LBFGS([W], max_iter=100, tolerance_grad=1e-10, tolerance_change=1e-14, history_size=50,
                         line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = th.nn.functional.cross_entropy(hf @ W.T, yf)
        loss.backward()
        return loss
    for _ in range(5):
        opt.step(closure)
    return W.detach().cpu().numpy()


print(f"ARC: accuracy {acc0:.5f}, cross-entropy fit set {ce(wo, hf, yf):.5f}, test set {ce(wo, ht, yt):.5f}")
for name, init in [("from ARC", wo.copy()), ("from zero", np.zeros_like(wo))]:
    W = fit(init)
    d = np.linalg.norm(cen(W) - cen(wo)) / np.linalg.norm(cen(wo))
    ratio = (cen(W) * cen(wo)).sum() / (cen(wo) ** 2).sum()
    print(f"\noptimum {name}: cross-entropy fit {ce(W, hf, yf):.5f}, test {ce(W, ht, yt):.5f}; accuracy change "
          f"{100 * (acc(W) - acc0):+.3f} points; same answer as ARC on {100 * ((ht @ th.as_tensor(W, device=dev).T).argmax(-1) == (ht @ th.as_tensor(wo, device=dev).T).argmax(-1)).double().mean().item():.2f}%")
    print(f"  relative distance from ARC (centred) {d:.4f}; size relative to ARC along ARC {ratio:.4f}")
    E = cen(W) - cen(wo)
    print(f"  largest differences: " + ", ".join(f"{j}->out{k} ARC {wo[k, j]:+.2f} diff {E[k, j]:+.3f}" for k, j in
                                                   sorted(np.ndindex(10, 16), key=lambda kj: -abs(E[kj]))[:8]))
    if name == "from ARC":
        np.save("optimal_readout.npy", W)

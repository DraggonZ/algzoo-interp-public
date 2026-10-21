"""APPROVED EXCEPTION (2026-10-18): the project rule forbids optimizing hand-built weights, even as a diagnostic;
this one test was approved. The hand-built networks themselves are unchanged; trained copies are a separate
experiment on what training does to a clean solution.
Question: does training starting from a hand-built network keep its close calls simple, or drift dense?
Setup: copies of v19 (16 neurons, kappa 0.5) and v21 (28 neurons). First exact rescaling (each hidden neuron's incoming
weights x c, outgoing x 1/c, so the function is unchanged) to give every neuron an activation RMS of 1, and one readout
scale chosen to minimise cross-entropy (argmax unchanged). Then AdamW (weight decay 0.001), batch 65,536, rates 1e-5 and
1e-4, 3,000 steps, float32 (a float64 run of 10,000 steps was too slow; partial log run_fp64_partial.log). At 0, 100, 1,000, 10,000 steps: accuracy, near ties lost to the third, lead maps at 3%
covering half of 100,000 lists, and the tree's share of the possible gain on near ties. ARC trained the same way for
reference (finetune/simpler.py: maps 9,040 -> 5,532). Predictions (written first): accuracy rises; the maps count rises
from about 50 toward thousands and the tree's gain falls, i.e. training drifts dense."""
import sys; sys.path.insert(0, "../../family")
import numpy as np, torch
from collections import Counter
from sklearn.ensemble import HistGradientBoostingClassifier
from common import DEV, dt, jac, top3, scores, T
from algzoo.handbuilt_v19 import build_v19
from algzoo.handbuilt_v21 import build_v21
g = torch.Generator(device=DEV).manual_seed(7); X = torch.randn(100_000, 10, device=DEV, dtype=dt, generator=g); l, s, t = top3(X); n = torch.arange(len(X), device=DEV)
G = torch.Generator(device=DEV).manual_seed(101); Y = torch.randn(3_000_000, 10, device=DEV, dtype=dt, generator=G); o = Y.argsort(1, descending=True); vv = Y.sort(1, descending=True).values
near = (vv[:, 1] > 0) & ((vv[:, 1] - vv[:, 2]) < 0.05 * vv[:, 1]); Y, o, vv = Y[near][:250000], o[near][:250000], vv[near][:250000]
F = np.c_[o[:, :3].cpu().numpy().astype(float), (vv[:, 2] / vv[:, 1]).cpu().numpy(), (vv[:, 0] / vv[:, 1]).cpu().numpy()]
def measure(W, lab):
    W = [w.detach().double().contiguous() for w in W]
    acc = (scores(W, X).argmax(1) == s).double().mean().item()
    J = jac(W, X); d = J[n, s] - J[n, t]; d = d / d.norm(dim=1, keepdim=True).clamp(min=1e-30)
    c = Counter(map(bytes, torch.round(d / 0.03).to(torch.int64).cpu().numpy())); cnt = np.array(sorted(c.values(), reverse=True)); half = np.searchsorted(np.cumsum(cnt) / len(X), 0.5) + 1
    y = (scores(W, Y).argmax(1) == o[:, 2]).cpu().numpy(); guess = max(y.mean(), 1 - y.mean())
    tr = (HistGradientBoostingClassifier(max_iter=300, max_leaf_nodes=63).fit(F[:200000], y[:200000]).predict(F[200000:]) == y[200000:]).mean()
    print(f"{lab:34s} accuracy {100*acc:5.2f}% | near ties lost to the third {100*y.mean():4.1f}% | lead maps covering half {half:6,} | tree's share of possible gain {100*(tr-guess)/max(1-guess,1e-9):3.0f}%", flush=True)
def balance(W):
    wi, wh, wo = [w.clone() for w in W]
    h = torch.zeros(len(X), len(wi), device=DEV, dtype=dt); H = []
    for k in range(10): h = torch.relu(h @ wh.T + X[:, k:k + 1] * wi); H.append(h)
    rms = torch.stack(H).pow(2).mean((0, 1)).sqrt().clamp(min=1e-12); c = 1 / rms
    wi = wi * c; wh = wh * c[:, None] / c[None, :]; wo = wo / c[None, :]
    W = [wi, wh, wo]; Sx = scores(W, X); best = None
    for temp in np.exp(np.linspace(np.log(1e-4), np.log(1e3), 120)):
        ce = torch.nn.functional.cross_entropy(Sx * temp, s).item()
        if best is None or ce < best[0]: best = (ce, temp)
    return [wi, wh, wo * best[1]]
def sc32(W, x):
    h = torch.zeros(len(x), W[0].shape[0], device=DEV, dtype=x.dtype)
    for k in range(10): h = torch.relu(h @ W[1].T + x[:, k:k + 1] * W[0])
    return h @ W[2].T
def train(W0, lr, lab):
    W = [w.clone().float().requires_grad_(True) for w in W0]; opt = torch.optim.AdamW(W, lr=lr, weight_decay=0.001); gt = torch.Generator(device=DEV).manual_seed(99)
    for step in range(1, 3001):
        x = torch.randn(65536, 10, device=DEV, dtype=torch.float32, generator=gt); loss = torch.nn.functional.cross_entropy(sc32(W, x), x.argsort(1)[:, -2])
        opt.zero_grad(); loss.backward(); opt.step()
        if step in (100, 1000, 3000): measure(W, f"{lab}, lr {lr:g}, {step} steps")
    return W
for name, W in (("hand-built v19", T(build_v19(kappa=0.5).params)), ("hand-built v21", T(build_v21()))):
    Wb = balance(W); measure(W, f"{name} as built"); measure(Wb, f"{name} rescaled (same function)")
    for lr in (1e-5, 1e-4): train(Wb, lr, name)

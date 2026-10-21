"""Concern: is 'dense' just 'imprecise weights'? Add random relative noise to the hand-built networks'
weights (each weight x (1 + s z), z standard normal; relative, so the networks' very different weight scales do not
matter) and measure the same density measures as for the trained networks, against accuracy. Measures: lead maps at
3% precision covering half of 100,000 lists; on near ties, the share where the network names the third, and a tree on
the top three (positions and two ratios): its accuracy and always-guess. If noisy hand-built networks at ARC's accuracy
look like ARC (thousands of maps, tree barely above guessing), density means imprecision; if they keep few maps and a
predictive tree, it does not. Prediction (written first): noise raises the maps count a lot (any noise breaks the
hand-built networks' exact structure) but the tree stays far more predictive than for ARC."""
import sys; sys.path.insert(0, "../../family")
import numpy as np, torch
from collections import Counter
from sklearn.ensemble import HistGradientBoostingClassifier
from common import DEV, dt, jac, top3, scores, T, ARC
from algzoo.handbuilt_v19 import build_v19
from algzoo.handbuilt_v21 import build_v21
g = torch.Generator(device=DEV).manual_seed(7); X = torch.randn(100_000, 10, device=DEV, dtype=dt, generator=g); l, s, t = top3(X); n = torch.arange(len(X), device=DEV)
G = torch.Generator(device=DEV).manual_seed(101); Y = torch.randn(3_000_000, 10, device=DEV, dtype=dt, generator=G); o = Y.argsort(1, descending=True); vv = Y.sort(1, descending=True).values
near = (vv[:, 1] > 0) & ((vv[:, 1] - vv[:, 2]) < 0.05 * vv[:, 1]); Y, o, vv = Y[near][:250000], o[near][:250000], vv[near][:250000]
F = np.c_[o[:, :3].cpu().numpy().astype(float), (vv[:, 2] / vv[:, 1]).cpu().numpy(), (vv[:, 0] / vv[:, 1]).cpu().numpy()]
def measure(W, lab):
    W = [w.contiguous() for w in W]
    acc = (scores(W, X).argmax(1) == s).double().mean().item()
    J = jac(W, X); d = J[n, s] - J[n, t]; d = d / d.norm(dim=1, keepdim=True).clamp(min=1e-30)
    c = Counter(map(bytes, torch.round(d / 0.03).to(torch.int64).cpu().numpy())); cnt = np.array(sorted(c.values(), reverse=True)); half = np.searchsorted(np.cumsum(cnt) / len(X), 0.5) + 1
    y = (scores(W, Y).argmax(1) == o[:, 2]).cpu().numpy(); guess = max(y.mean(), 1 - y.mean())
    tr = (HistGradientBoostingClassifier(max_iter=300, max_leaf_nodes=63).fit(F[:200000], y[:200000]).predict(F[200000:]) == y[200000:]).mean()
    print(f"{lab:30s} accuracy {100*acc:5.2f}% | near ties lost to the third {100*y.mean():4.1f}% | maps covering half {half:6,} | tree {100*tr:5.1f}% vs guess {100*guess:5.1f}% (unexplained {100*(1-tr):4.1f}%)", flush=True)
measure(ARC, "ARC (trained)")
for name, W0 in (("hand-built v21 (28)", T(build_v21())), ("hand-built v19 (16)", T(build_v19(kappa=0.5).params))):
    measure(W0, f"{name}, no noise")
    for sd in (0.001, 0.003, 0.01, 0.03):
        for rep in range(2):
            gn = torch.Generator(device=DEV).manual_seed(1000 + rep)
            W = [w * (1 + sd * torch.randn(w.shape, device=DEV, dtype=dt, generator=gn)) for w in W0]
            measure(W, f"{name}, noise {sd:g} #{rep}")

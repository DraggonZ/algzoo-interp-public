"""Close-call density of v23 variants, measured exactly as in siblings/siblings.py: accuracy on 100,000 lists, share of
near ties (third within 5% of the second) where it names the third, second-over-third lead maps at 3% precision covering
half of the 100,000 lists, and the share of the possible gain over always-guessing that a tree on the top three
(positions and two ratios) reaches on near ties."""
import sys; sys.path.insert(0, "../../family")
import numpy as np, torch
from collections import Counter
from sklearn.ensemble import HistGradientBoostingClassifier
from common import DEV, dt, jac, top3, scores, T, ARC
from algzoo.handbuilt_v23 import build_v23
from algzoo.handbuilt_v19 import build_v19
g = torch.Generator(device=DEV).manual_seed(7); X = torch.randn(100_000, 10, device=DEV, dtype=dt, generator=g); l, s, t = top3(X); n = torch.arange(len(X), device=DEV)
G = torch.Generator(device=DEV).manual_seed(101); Y = torch.randn(3_000_000, 10, device=DEV, dtype=dt, generator=G); o = Y.argsort(1, descending=True); vv = Y.sort(1, descending=True).values
near = (vv[:, 1] > 0) & ((vv[:, 1] - vv[:, 2]) < 0.05 * vv[:, 1]); Y, o, vv = Y[near][:250000], o[near][:250000], vv[near][:250000]
F = np.c_[o[:, :3].cpu().numpy().astype(float), (vv[:, 2] / vv[:, 1]).cpu().numpy(), (vv[:, 0] / vv[:, 1]).cpu().numpy()]
def measure(name, W):
    W = [w.contiguous() for w in W]
    acc = (scores(W, X).argmax(1) == s).double().mean().item()
    J = jac(W, X); d = J[n, s] - J[n, t]; d = d / d.norm(dim=1, keepdim=True).clamp(min=1e-30)
    c = Counter(map(bytes, torch.round(d / 0.03).to(torch.int64).cpu().numpy())); cnt = np.array(sorted(c.values(), reverse=True)); half = np.searchsorted(np.cumsum(cnt) / len(X), 0.5) + 1
    y = (scores(W, Y).argmax(1) == o[:, 2]).cpu().numpy(); guess = max(y.mean(), 1 - y.mean())
    clf = HistGradientBoostingClassifier(max_iter=300, max_leaf_nodes=63).fit(F[:200000], y[:200000]); tr = (clf.predict(F[200000:]) == y[200000:]).mean()
    gain = (tr - guess) / max(1 - guess, 1e-9)
    print(f"{name:34s} accuracy {100*acc:5.2f}% | third wins {100*y.mean():4.1f}% of near ties | lead maps covering half {half:6,} | tree {100*tr:4.1f}% vs guess {100*guess:4.1f}%, {100*gain:3.0f}% of the possible gain", flush=True)
    return acc, y.mean(), half, gain
if __name__ == "__main__":
    nets = {"ARC": ARC, "v19 (kappa 0.5)": T(build_v19(kappa=0.5).params)}
    for w in (0.0, 0.25, 1.0): nets[f"v23, loop-back {w}"] = T(build_v23(w=w).params)
    nets["v23, loop-back 0, v18's wave"] = T(build_v23(shape="symmetric").params)
    for k, W in nets.items(): measure(k, W)

"""P6: close-call density, measured exactly as superposition/siblings/siblings.py (same lists, seeds and tree)."""
import sys; sys.path.insert(0, "/data/output/research/algzoo/20261017_handbuilt/family")
import numpy as np, torch
from collections import Counter
from sklearn.ensemble import HistGradientBoostingClassifier
from common import DEV, dt, jac, top3, scores, T, ARC
from algzoo.handbuilt_v19 import build_v19
from algzoo.handbuilt_v23_from_v19 import build_v23f

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
    print(f"{lab:38s} accuracy {100*acc:5.2f}% | near ties lost to the third {100*y.mean():4.1f}% | maps covering half {half:6,} | tree {100*tr:5.1f}% vs guess {100*guess:5.1f}% (share of possible gain {100*(tr-guess)/max(1-guess,1e-9):3.0f}%)", flush=True)
    return acc, y.mean(), half, tr, guess


if __name__ == "__main__":
    measure(ARC, "ARC (trained)")
    measure(T(build_v19(kappa=0.5).params), "hand-built v19")
    measure(T(build_v23f().params), "v23 from v19, loop gain 1e-4")
    measure(T(build_v23f(ep=0.3).params), "v23 from v19, loop gain 0.3")

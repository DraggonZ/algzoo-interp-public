"""Idealized scores (no network): does a keep gate relative to the runner-up so far make near ties ARC-like?
Votes: alpha x + beta g, with g = relu(x - theta m2) (ARC's keep rule, theta 0.75; m2 the runner-up before x), or the
variant a runner-up comparator gives for free, g = (1 - theta) x when x is above m2. Largest hidden. Optional fading of
votes (rho per step of age, uncompensated). Measures as in siblings.py: accuracy, third wins at near ties, lead maps,
tree on the top three. Prediction (written first): the rule alone gives third wins well above v23's 1.4% and a tree far
less predictive (the runner-up at a number's arrival is often outside the top three)."""
import sys; sys.path.insert(0, "/data/output/research/algzoo/20261017_handbuilt/family")
import numpy as np, torch
from collections import Counter
from sklearn.ensemble import HistGradientBoostingClassifier
from common import DEV, dt, top3
def score_fn(kind, theta=0.75, alpha=0.5, beta=0.5, rho=1.0):
    def f(X):
        m1 = torch.zeros(len(X), device=DEV, dtype=dt); m2 = torch.zeros_like(m1); V = []
        for t in range(10):
            x = X[:, t]
            if kind == "positive": g = torch.relu(x)
            elif kind == "runner-up": g = torch.relu(x - theta * m2)
            else: g = torch.where(x > m2, (1 - theta) * torch.relu(x), torch.relu(x - theta * m2))
            V.append((alpha * x + beta * g) * rho ** (9 - t))
            m2 = torch.maximum(m2, torch.minimum(x, m1)); m1 = torch.maximum(m1, x)
        V = torch.stack(V, 1); isL = torch.nn.functional.one_hot(X.argmax(1), 10).to(dt)
        return V - 10 * isL * X.max(1).values[:, None].abs()
    return f
g = torch.Generator(device=DEV).manual_seed(7); X = torch.randn(100_000, 10, device=DEV, dtype=dt, generator=g); l, s, t = top3(X); n = torch.arange(len(X), device=DEV)
G = torch.Generator(device=DEV).manual_seed(101); Y = torch.randn(3_000_000, 10, device=DEV, dtype=dt, generator=G); o = Y.argsort(1, descending=True); vv = Y.sort(1, descending=True).values
near = (vv[:, 1] > 0) & ((vv[:, 1] - vv[:, 2]) < 0.05 * vv[:, 1]); Y, o, vv = Y[near][:250000], o[near][:250000], vv[near][:250000]
F = np.c_[o[:, :3].cpu().numpy().astype(float), (vv[:, 2] / vv[:, 1]).cpu().numpy(), (vv[:, 0] / vv[:, 1]).cpu().numpy()]
def measure(name, f):
    Xg = X.clone().requires_grad_(True); sc = f(Xg); lead = sc[n, s] - sc[n, t]; d, = torch.autograd.grad(lead.sum(), Xg)
    acc = (sc.argmax(1) == s).double().mean().item(); d = d / d.norm(dim=1, keepdim=True).clamp(min=1e-30)
    cnt = np.array(sorted(Counter(map(bytes, torch.round(d / 0.03).to(torch.int64).cpu().numpy())).values(), reverse=True)); half = np.searchsorted(np.cumsum(cnt) / len(X), 0.5) + 1
    y = (f(Y).argmax(1) == o[:, 2]).cpu().numpy(); guess = max(y.mean(), 1 - y.mean())
    tr = (HistGradientBoostingClassifier(max_iter=300, max_leaf_nodes=63).fit(F[:200000], y[:200000]).predict(F[200000:]) == y[200000:]).mean()
    print(f"{name:40s} accuracy {100*acc:5.1f}% | third wins {100*y.mean():4.1f}% | maps {half:6,} | tree {100*tr:4.1f}% vs guess {100*guess:4.1f}%, {100*(tr-guess)/max(1-guess,1e-9):3.0f}% of gain", flush=True)
if __name__ == "__main__":
    measure("keep positives (v23)", score_fn("positive"))
    measure("keep relative to runner-up, theta 0.75", score_fn("runner-up"))
    measure("comparator variant, theta 0.75", score_fn("comparator"))
    for rho in (0.97, 0.94):
        measure(f"runner-up rule + fading {rho} uncompensated", score_fn("runner-up", rho=rho))

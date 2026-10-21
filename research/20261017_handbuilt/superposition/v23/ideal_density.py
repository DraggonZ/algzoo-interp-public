"""Idealized score models (no network): count second-over-third lead maps at 3% precision covering half of 100,000
random lists, as in siblings.py, for vote mechanisms written as formulas. Scores are piecewise linear in the inputs, so
autograd gives the exact lead map per list."""
import numpy as np, torch
from collections import Counter
DEV = "cuda"; dt = torch.float64
g = torch.Generator(device=DEV).manual_seed(7); X = torch.randn(100_000, 10, device=DEV, dtype=dt, generator=g)
o = X.argsort(1, descending=True); L, S, T = o[:, 0], o[:, 1], o[:, 2]; n = torch.arange(len(X), device=DEV)
pos = torch.arange(10, device=DEV, dtype=dt)
def kernel(angles):
    th = torch.tensor(np.radians(angles), device=DEV, dtype=dt); nk = torch.tensor([1. if a in (0, 180) else 2. for a in angles], device=DEV, dtype=dt)
    d = pos[:, None] - pos[None, :]; return (nk * torch.cos(th * d[..., None])).sum(-1) / nk.sum()
KARC = kernel([0, 43, 83, 125, 180])
def run(X):
    m1 = torch.zeros(len(X), device=DEV, dtype=dt); m2 = torch.zeros_like(m1); M1, M2 = [], []
    for t in range(10):  # running largest and second before step t (start at 0, as a bias-free network does)
        M1.append(m1); M2.append(m2); x = X[:, t]
        m2 = torch.maximum(m2, torch.minimum(x, m1)); m1 = torch.maximum(m1, x)
    return torch.stack(M1, 1), torch.stack(M2, 1), m1
def model(name, X):
    M1, M2, m1 = run(X); rec = X > M1; isL = torch.nn.functional.one_hot(X.argmax(1), 10).to(dt)
    kept = (X >= 0.75 * M2) & (X > 0)
    if name == "sharp, all numbers":                v = X
    elif name == "sharp, keep or forget":           v = X * kept
    elif name == "sharp, imprint at write time":    v = X - 0.9 * torch.maximum(M1, X)
    elif name == "broad (ARC angles), all numbers": v = X @ KARC.T
    elif name == "broad (ARC angles), keep or forget": v = (X * kept) @ KARC.T
    elif name == "broad (ARC angles), positives":   v = torch.relu(X) @ KARC.T
    return v - 5 * isL * m1[:, None] - (isL * v).detach() * 0  # hide the largest's position (exact, sharp)
def maps(name):
    Xg = X.clone().requires_grad_(True); sc = model(name, Xg); lead = sc[n, S] - sc[n, T]
    d, = torch.autograd.grad(lead.sum(), Xg); acc = (sc.argmax(1) == S).double().mean().item()
    d = d / d.norm(dim=1, keepdim=True).clamp(min=1e-30)
    c = Counter(map(bytes, torch.round(d / 0.03).to(torch.int64).cpu().numpy())); cnt = np.array(sorted(c.values(), reverse=True))
    return acc, np.searchsorted(np.cumsum(cnt) / len(X), 0.5) + 1
if __name__ == "__main__":
    print("kernel K(0..9) with ARC's angles:", " ".join(f"{v:.2f}" for v in KARC[0].tolist()))
    for name in ("sharp, all numbers", "sharp, keep or forget", "sharp, imprint at write time", "broad (ARC angles), all numbers",
                 "broad (ARC angles), positives", "broad (ARC angles), keep or forget"):
        acc, h = maps(name); print(f"{name:38s} accuracy {100*acc:5.1f}%  lead maps covering half {h:6,}")

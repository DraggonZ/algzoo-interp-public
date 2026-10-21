"""Share of ARC's wrong lists on which other networks (and a noisy perfect solver) make the same wrong choice."""
from common import *
P = arc(); g = gen(9303)
x = draw(1_000_000, g); y = target(x); pr = predict(P, x)
accA = (pr == y).double().mean().item(); w = pr != y
print(f"ARC accuracy {100*accA:.2f}%, wrong lists {w.sum().item()}")
res = {}
for s, Q in others().items():
    q = predict(Q, x)
    res[s] = ((q == pr) & w).sum().item() / w.sum().item(), (q == y).double().mean().item()
for s, (sh, acc) in res.items():
    print(f"seed {s}: accuracy {100*acc:.2f}%, same wrong choice on ARC's mistakes {100*sh:.1f}%")
v = np.array([r[0] for r in res.values()])
print(f"32 networks: median {100*np.median(v):.1f}%, min {100*v.min():.1f}%, max {100*v.max():.1f}%")

def noisy(xx, s, gg):
    return target(xx + s * torch.randn(xx.shape, dtype=DT, device=DEV, generator=gg))
# calibrate s by bisection on separate lists
xc = draw(2_000_000, g); yc = target(xc); lo, hi = 0.0, 0.5
for _ in range(30):
    m = (lo + hi) / 2
    a = (noisy(xc, m, g) == yc).double().mean().item()
    lo, hi = (m, hi) if a > accA else (lo, m)
s = (lo + hi) / 2
nn = noisy(x, s, g)
print(f"imprecise solver: s = {s:.4f}, accuracy {100*(nn==y).double().mean().item():.2f}%, "
      f"same wrong choice on ARC's mistakes {100*((nn==pr)&w).sum().item()/w.sum().item():.1f}%")

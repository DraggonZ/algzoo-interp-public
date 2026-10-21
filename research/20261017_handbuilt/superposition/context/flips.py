"""What context decides a near tie between the second and third? On near ties (third within 5% of the second, second
positive), change one context variable and count how often the second-versus-third decision flips. Smaller numbers are
redrawn from their own law (normal truncated below the third), so every list keeps its top three."""
import sys; sys.path.insert(0, "../steer")
import numpy as np
from harness import *
from scipy.stats import norm
g = torch.Generator(device=DEV).manual_seed(101)
X = lists(6_000_000, seed=101); o = X.argsort(1, descending=True); v = X.sort(1, descending=True).values
near = (v[:, 1] > 0) & ((v[:, 1] - v[:, 2]) < 0.05 * v[:, 1]); X, o = X[near], o[near]
L, S, T = o[:, 0], o[:, 1], o[:, 2]; n = torch.arange(len(X), device=DEV); xT = X[n, T]
small = torch.ones_like(X, dtype=torch.bool); small[n, L] = small[n, S] = small[n, T] = False
def redraw(mask):
    u = torch.rand(X.shape, device=DEV, dtype=dt, generator=g) * torch.tensor(norm.cdf(xT.cpu().numpy()), device=DEV, dtype=dt)[:, None]
    new = torch.tensor(norm.ppf(u.clamp(1e-12, 1 - 1e-12).cpu().numpy()), device=DEV, dtype=dt)
    Y = X.clone(); Y[mask] = new[mask]; return Y
d0 = run(ARC, X).argmax(1); keep = (d0 == S) | (d0 == T)
def flip(Y):
    d = run(ARC, Y).argmax(1); return 100 * (((d == S) != (d0 == S)) & keep).double().sum().item() / keep.double().sum().item()
print(f"{len(X):,} near ties; network names the third on {100*(d0 == T).double().mean().item():.1f}%")
print(f"redraw all seven smaller numbers: decision flips on {flip(redraw(small)):.1f}% (repeat {flip(redraw(small)):.1f}%)")
print(f"set the smaller numbers to their mean: {flip(torch.where(small, torch.tensor(-norm.pdf(xT.cpu().numpy())/norm.cdf(xT.cpu().numpy()), device=DEV, dtype=dt)[:, None], X)):.1f}%")
E, Lt = torch.minimum(S, T), torch.maximum(S, T)
pos = torch.arange(10, device=DEV)[None].expand(len(X), 10)
for lab, sel in (("right after the earlier contender", pos == (E + 1)[:, None]), ("right after the later contender", pos == (Lt + 1)[:, None]),
                 ("right after the largest", pos == (L + 1)[:, None]), ("right before the earlier contender", pos == (E - 1)[:, None]),
                 ("between the contenders", (pos > E[:, None]) & (pos < Lt[:, None])), ("before everything", pos < torch.minimum(E, L)[:, None]),
                 ("after everything", pos > torch.maximum(Lt, L)[:, None])):
    m = sel & small; cnt = m.any(1)
    print(f"redraw the smaller numbers {lab:36s} (present in {100*cnt.double().mean().item():3.0f}% of lists, {m.double().sum(1)[cnt].mean().item():.1f} per list): flips {flip(redraw(m)):5.1f}%")
# largest moved to another slot (swap with a smaller number's slot), all values kept
Y = X.clone(); r = torch.randint(0, 10, (len(X),), device=DEV, generator=g)
ok = small[n, r]; Y[n[ok], L[ok]], Y[n[ok], r[ok]] = X[n[ok], r[ok]], X[n[ok], L[ok]]
print(f"largest moved to a random smaller number's slot (swapped; {100*ok.double().mean().item():.0f}% of lists): flips {flip(Y):.1f}% (counted over all lists)")
for f in (0.9, 1.1):
    Y = X.clone(); Y[n, L] *= f; print(f"largest's value x{f}: flips {flip(Y):.1f}%")

"""Morph-and-average, after Obermeyer et al. (Nature 2026), applied to inputs (every list is a valid input, so no
generative model is needed). Near ties the network gets right (names the second): move only the seven smaller numbers
along the gradient of (score of third - score of second), in small unit-length steps of 0.01 x second, staying below
the third, until the network names the third. Record the morph d (per unit of the second). Then: (1) the average morph
aligned by position relative to the earlier contender, the later one, and the largest; (2) consistency: within each
arrangement (positions of largest, second, third), the cosine between the mean morphs of two random halves of its
lists, and the mean cosine between single morphs."""
import sys; sys.path.insert(0, "../steer")
import numpy as np
from harness import *
X0 = lists(3_000_000, seed=401); o = X0.argsort(1, descending=True); v = X0.sort(1, descending=True).values
near = (v[:, 1] > 0.3) & ((v[:, 1] - v[:, 2]) < 0.05 * v[:, 1])
X0, o = X0[near], o[near]; ok = run(ARC, X0).argmax(1) == o[:, 1]; X0, o = X0[ok], o[ok]
L, S, T = o[:, 0], o[:, 1], o[:, 2]; n = torch.arange(len(X0), device=DEV); xs, xt = X0[n, S], X0[n, T]
small = torch.ones_like(X0, dtype=torch.bool); small[n, L] = small[n, S] = small[n, T] = False
Y = X0.clone(); done = torch.zeros(len(X0), dtype=torch.bool, device=DEV); D = torch.zeros_like(X0)
for it in range(300):
    Yg = Y.detach().requires_grad_(True); sc = scores(ARC, Yg); obj = (sc[n, T] - sc[n, S]).sum(); g, = torch.autograd.grad(obj, Yg)
    g = g * small; g = g / g.norm(dim=1, keepdim=True).clamp(min=1e-12)
    step = (0.01 * xs)[:, None] * g * (~done)[:, None]
    Y = Y.detach() + step; Y = torch.where(small, torch.min(Y, (xt - 1e-6)[:, None]), X0)
    flip = (scores(ARC, Y).argmax(1) == T) & ~done; D[flip] = ((Y - X0) / xs[:, None])[flip]; done |= flip
    if done.all(): break
print(f"{len(X0):,} right near ties; flipped to the third within 300 steps: {100*done.double().mean().item():.1f}%; median morph length {D[done].norm(dim=1).median().item():.3f} x second")
d, m = D[done], done
Lm, Sm, Tm = L[m], S[m], T[m]; E, Lt = torch.minimum(Sm, Tm), torch.maximum(Sm, Tm); third_late = Tm > Sm
pos = torch.arange(10, device=DEV)[None].expand_as(d); sm = small[m]
def by(offset, lab):
    row = []
    for k in range(-9, 10):
        sel = sm & (offset == k)
        if sel.sum() > 2000: row.append(f"{k:+d}:{d[sel].mean().item():+.3f}")
    print(f"mean morph by offset from {lab}: " + " ".join(row))
for lab, ref in (("the second", Sm), ("the third", Tm), ("the largest", Lm)):
    by(pos - ref[:, None], lab)
print(f"share of the morph's length on numbers before vs after the earlier contender: before {(d.abs() * (sm & (pos < E[:, None]))).sum().item() / d.abs().sum().item():.2f}, between {(d.abs() * (sm & (pos > E[:, None]) & (pos < Lt[:, None]))).sum().item() / d.abs().sum().item():.2f}, after {(d.abs() * (sm & (pos > Lt[:, None]))).sum().item() / d.abs().sum().item():.2f}")
# consistency within arrangements
key = (Lm * 100 + Sm * 10 + Tm).cpu().numpy(); dn = (d / d.norm(dim=1, keepdim=True)).cpu().numpy()
rng = np.random.default_rng(0); halves, singles = [], []
for k in np.unique(key):
    idx = np.where(key == k)[0]
    if len(idx) < 40: continue
    rng.shuffle(idx); a, b = idx[: len(idx) // 2], idx[len(idx) // 2:]
    ma, mb = dn[a].mean(0), dn[b].mean(0); halves.append(ma @ mb / np.linalg.norm(ma) / np.linalg.norm(mb))
    sub = dn[idx[:40]]; c = sub @ sub.T; singles.append((c.sum() - 40) / (40 * 39))
print(f"{len(halves)} arrangements: split-half cosine of mean morphs median {np.median(halves):.2f} (10-90%: {np.quantile(halves, 0.1):.2f}-{np.quantile(halves, 0.9):.2f}); mean cosine between single morphs median {np.median(singles):.2f}")

"""How far is each right answer from being flipped by the context alone? Keep the top three fixed and move only the
seven smaller numbers, each by at most r (absolute), staying below the third. To first order the lead of the second
over competitor k changes by sum_j (c_kj) d_j, so the smallest r that can flip it is lead_k / sum_j |c_kj| over the
smaller numbers (minimum over k). Verify by applying that worst-case move (times 1.05) on the real network. Report r as
a fraction of the second's value, by the second-third gap."""
import sys; sys.path.insert(0, "../steer")
import numpy as np
from harness import *
X = lists(300_000, seed=301); o = X.argsort(1, descending=True); L, S, T = o[:, 0], o[:, 1], o[:, 2]; v = X.sort(1, descending=True).values
n = torch.arange(len(X), device=DEV); a0 = run(ARC, X).argmax(1); ok = (a0 == S) & (v[:, 1] > 0.2)
X, L, S, T, v, n = X[ok], L[ok], S[ok], T[ok], v[ok], torch.arange(int(ok.sum()), device=DEV)
J = jac(ARC, X); sc = torch.einsum("noi,ni->no", J, X)
small = torch.ones_like(X, dtype=torch.bool); small[n, L] = small[n, S] = small[n, T] = False
lead = sc[n, S][:, None] - sc                                 # lead over each k
Cd = J[n, S][:, None, :] - J                                   # coefficient differences, n x k x inputs
den = (Cd.abs() * small[:, None, :]).sum(2); r_k = lead / den.clamp(min=1e-12); r_k[n, S] = 1e9
r, kmin = r_k.min(1)
# verify: move each smaller number by -sign(Cd) * 1.05 r toward the competitor, clipped below the third
d = -torch.sign(Cd[n, kmin]) * (1.05 * r)[:, None] * small
Y = torch.minimum(X + d, (X[n, T] - 1e-6)[:, None].expand_as(X)); Y = torch.where(small, Y, X)
flipped = run(ARC, Y).argmax(1) != S
rr = r / X[n, S]; gap = (v[:, 1] - v[:, 2]) / v[:, 1]
print(f"{len(X):,} right lists; the worst-case move of the smaller numbers by 1.05 r really flips {100*flipped.double().mean().item():.0f}%")
for lo, hi in ((0, 0.05), (0.05, 0.1), (0.1, 0.2), (0.2, 0.5), (0.5, 1.01)):
    m = (gap >= lo) & (gap < hi)
    q = torch.quantile(rr[m], torch.tensor([0.1, 0.5, 0.9], device=DEV, dtype=dt)).cpu().numpy()
    print(f"second-third gap {lo:>4}-{hi:<4}: flip radius per unit of the second, 10/50/90%: {np.round(q, 3)} (n {int(m.sum()):,})")
print(f"radius compared with the gap itself (median of radius/gap): {torch.median(rr / gap.clamp(min=1e-6)).item():.2f}")

"""Which share of each contender's vote passes through the detector? The final lead D is exactly linear in the inputs on
each list: D = sum_i c_i x_i. Split c_i x_i into the part that passes through neuron 5 at some step (first order:
the change in D when neuron 5 is scaled, attributed to input i) and the rest. If the second and third send different
shares through the detector, scaling it moves the lead."""
import sys; sys.path.insert(0, "../../family")
import torch
from common import *
from memory_gain import X, l, s, t3
W = ARC; Xm = X[:50000].clone().requires_grad_(True); sm, tm, lm = s[:50000], t3[:50000], l[:50000]; n = torch.arange(len(Xm), device=DEV)
h = torch.zeros(len(Xm), 16, device=DEV, dtype=dt); Hs, Zs = [], []
for t in range(10):
    z = h @ W[1].T + Xm[:, t:t + 1] * W[0]; Zs.append(z); h = torch.relu(z); Hs.append(h)
sc = h @ W[2].T; D = sc[n, sm] - sc[n, tm]
c, = torch.autograd.grad(D.sum(), Xm, retain_graph=True)                 # exact coefficients c_i
# path through neuron 5: scale it by (1+a): dD/da = sum_t h5_t * dD/dh5_t ; attribute h5_t to inputs via its own linear map
GZ = torch.autograd.grad(D.sum(), Zs, retain_graph=True)
via = torch.zeros_like(Xm)
for t in range(10):
    gh5 = (GZ[t + 1] @ W[1])[:, 5] if t < 9 else (W[2][sm, 5] - W[2][tm, 5])   # dD/dh5_t
    J5, = torch.autograd.grad(Hs[t][:, 5].sum(), Xm, retain_graph=True)          # h5_t = J5 . x
    via += gh5[:, None] * J5 * Xm.detach()
cx = c * Xm.detach()
fs = via[n, sm] / cx[n, sm]; ft = via[n, tm] / cx[n, tm]
print(f"near-tie lists {len(Xm):,}. Share of each contender's part of the lead that passes through the detector (median, 10-90%):")
q = lambda v: f"{v.median():+.2f} ({v.quantile(0.1):+.2f} to {v.quantile(0.9):+.2f})"
print(f"   second: {q(fs)} | third: {q(ft)}")
for name, mask in (("second before third", sm < tm), ("third before second", tm < sm), ("second right after the largest", sm == lm + 1), ("third right after the largest", tm == lm + 1), ("second at step 9", sm == 9), ("third at step 9", tm == 9)):
    print(f"   {name:32s} n {int(mask.sum()):6,}: second {fs[mask].median():+.2f}, third {ft[mask].median():+.2f}")
print(f"largest's part through the detector, median share {(via[n, lm] / cx[n, lm]).median():+.2f}; sum over inputs of the detector path, median |.| {via.sum(1).abs().median():.2f} against |D| median {D.abs().median():.2f}")

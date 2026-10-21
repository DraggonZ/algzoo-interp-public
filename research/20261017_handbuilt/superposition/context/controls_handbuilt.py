"""Control (2026-10-19): are the post's two close-call measures special to ARC, or what any network shows on
near ties? Same near ties and definitions as units.py and flips.py, run on ARC and the hand-built networks.
(a) units.py: summed |first-order effect| of all (neuron, step) units on the second-vs-third lead, over |lead|; share in
the top unit; units whose 5% nudge alone exceeds the lead. (b) flips.py: redraw all seven smaller numbers (top three
kept), share of second-vs-third decisions that flip."""
import sys; sys.path.insert(0, "../steer")
import numpy as np
from harness import *
from scipy.stats import norm
from algzoo.handbuilt_v19 import build_v19
from algzoo.handbuilt_v21 import build_v21
from algzoo.handbuilt_v23 import build_v23
Tw = lambda P: [torch.tensor(np.asarray(p, float), device=DEV, dtype=dt) for p in P]
NETS = {"ARC": ARC, "twin16 (v23, ARC's constants)": Tw(build_v23(U_mem=1.0, share=0.25, rho=0.99, level_exact=False).params),
        "v19 (16)": Tw(build_v19(kappa=0.5).params), "v21 (28)": Tw(build_v21())}
X = lists(6_000_000, seed=101); o = X.argsort(1, descending=True); v = X.sort(1, descending=True).values
near = (v[:, 1] > 0) & ((v[:, 1] - v[:, 2]) < 0.05 * v[:, 1]); X, o = X[near], o[near]
L, S, T = o[:, 0], o[:, 1], o[:, 2]; n = torch.arange(len(X), device=DEV); xT = X[n, T]
small = torch.ones_like(X, dtype=torch.bool); small[n, L] = small[n, S] = small[n, T] = False
def redraw(mask, seed):
    g = torch.Generator(device=DEV).manual_seed(seed)
    u = torch.rand(X.shape, device=DEV, dtype=dt, generator=g) * torch.tensor(norm.cdf(xT.cpu().numpy()), device=DEV, dtype=dt)[:, None]
    new = torch.tensor(norm.ppf(u.clamp(1e-12, 1 - 1e-12).cpu().numpy()), device=DEV, dtype=dt)
    Y = X.clone(); Y[mask] = new[mask]; return Y
Y1, Y2 = redraw(small, 7), redraw(small, 8)
print(f"{len(X):,} near ties (third within 5% of the second, second positive)")
for name, W in NETS.items():
    d0 = run(W, X).argmax(1); keep = (d0 == S) | (d0 == T)
    fl = [100 * (((run(W, Y).argmax(1) == S) != (d0 == S)) & keep).double().sum().item() / keep.double().sum().item() for Y in (Y1, Y2)]
    Xs = X[:100000].clone().requires_grad_(True); nn = torch.arange(len(Xs), device=DEV); H = W[0].shape[0]
    hh = torch.zeros(len(Xs), H, device=DEV, dtype=dt); leaves = []
    for t in range(10): hh = torch.relu(hh @ W[1].T + Xs[:, t:t + 1] * W[0]); leaves.append(hh)
    Sc = hh @ W[2].T; lead = Sc[nn, S[:100000]] - Sc[nn, T[:100000]]
    G = torch.autograd.grad(lead.sum(), leaves)
    E = torch.stack([(G[t] * leaves[t]).detach() for t in range(10)], 1).reshape(len(Xs), -1).abs(); tot = E.sum(1)
    ok = lead.abs() > 0
    ratio = torch.median(tot[ok] / lead.abs()[ok]).item(); top1 = torch.median(E.max(1).values[ok] / tot[ok]).item()
    r = (E > 20 * lead.abs()[:, None]).sum(1).double()[ok]
    print(f"{name:32s} names the third {100*(d0 == T).double().mean().item():4.1f}% | redraw seven: flips {fl[0]:4.1f}% / {fl[1]:4.1f}% | "
          f"total |effect| / |lead| median {ratio:8.0f} | top unit's share {top1:.2f} | units that could flip alone: median {r.median().item():.0f} of {10*H} "
          f"| exact ties (lead 0) {100*(~ok).double().mean().item():.2f}%", flush=True)

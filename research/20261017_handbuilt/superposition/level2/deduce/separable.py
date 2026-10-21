"""Do the smaller numbers act one at a time? Per arrangement (canonical largest three, others at their mean), record the
network's exact score vector as each smaller slot alone moves over a 1-D grid of values (others canonical). Model a list
as: canonical scores + the top three's deviation through the canonical map J + the sum of each smaller number's own
(interpolated) change. If this additive model matches the network, the smaller numbers act independently and only cost
stands in the way; if not, they interact."""
import sys; sys.path.insert(0, "../../../family")
import numpy as np, torch
from scipy.stats import norm
from common import *
from case_maps import A, idx, C, v1, v2, v3, rest
G = torch.linspace(-3.0, v3, 25, device=DEV, dtype=dt)
def build(W):
    J = jac(W, C); S0 = scores(W, C)                     # 720 x 10 x 10, 720 x 10
    D = torch.zeros(720, 10, len(G), 10, device=DEV, dtype=dt)   # arrangement, slot, grid value, score
    for slot in range(10):
        X = C.repeat_interleave(len(G), 0).clone(); X[:, slot] = G.repeat(720)
        D[:, slot] = (scores(W, X).reshape(720, len(G), 10) - S0[:, None])
    return J, S0, D
def predict(W, X, J, S0, D, additive=True):
    l, s, t = top3(X); a = torch.tensor([idx[q] for q in zip(l.tolist(), s.tolist(), t.tolist())], device=DEV)
    dx = X - C[a]; top = torch.zeros_like(X, dtype=torch.bool); top[torch.arange(len(X)), l] = top[torch.arange(len(X)), s] = top[torch.arange(len(X)), t] = True
    out = S0[a] + torch.einsum("noi,ni->no", J[a], dx * top)
    if additive:
        pos = ((X - G[0]) / (G[1] - G[0])).clamp(0, len(G) - 1.0001); i0 = pos.floor().long(); w = (pos - i0)[..., None]
        n = torch.arange(len(X), device=DEV)[:, None]; sl = torch.arange(10, device=DEV)[None]
        Dv = D[a[:, None], sl, i0] * (1 - w) + D[a[:, None], sl, i0 + 1] * w          # n x 10 slots x 10 scores
        out = out + (Dv * (~top)[..., None]).sum(1)
    else:
        out = out + torch.einsum("noi,ni->no", J[a], dx * ~top)
    return out.argmax(1), s
if __name__ == "__main__":
    gg = torch.Generator(device=DEV).manual_seed(3); X = torch.randn(500_000, 10, device=DEV, dtype=dt, generator=gg)
    for name, W in [("ARC", ARC)] + [(f"seed {s}", trained(s)) for s in (100, 104, 107, 111, 114, 209, 211)]:
        J, S0, D = build(W); tr = (scores(W, X).argmax(1) == top3(X)[1]); r = []
        for lab, add in (("linear", False), ("additive", True)):
            p, s = predict(W, X, J, S0, D, add); ok = p == s
            r.append(f"{lab} {100*(ok.double().mean()-tr.double().mean()).item():+6.2f}")
        print(f"{name:9s} true {100*tr.double().mean().item():.2f}% | error, points: " + " | ".join(r) + f"  (additive costs {720*(1+10+10*len(G)):,} runs)", flush=True)

"""Sensitive weight directions from near-miss lead gradients; restore rounded weights along them."""
from common import *
from torch.func import jacrev, vmap
P = arc(); g = gen(9505); f0 = flat(P)

def sc1(fv, xi):
    return scores(unflat(fv, P), xi[None])[0]

# vote per unit v: median over lists of d score_p / d x_p, positions p with x_p > 0 and not the largest
xv = draw(20_000, g)
J = torch.cat([vmap(jacrev(sc1, argnums=1), in_dims=(None, 0))(f0, xv[i:i+2000]) for i in range(0, 20000, 2000)])  # [B,10,10]
diag = torch.diagonal(J, dim1=1, dim2=2)
mask = (xv > 0) & (torch.arange(10, device=DEV)[None] != xv.argmax(1)[:, None])
v = diag[mask].median().item()
print(f"vote per unit v = {v:.4f} (from {mask.sum().item()} positive non-largest numbers)")

# M = sum 1[lead_j < 2v] g_j g_j^T
x = draw(20_000, g); y = target(x)
M = torch.zeros(432, 432, dtype=DT, device=DEV); nsel = 0
for i in range(0, 20000, 1000):
    xb, yb = x[i:i+1000], y[i:i+1000]
    S = scores(P, xb)
    G = vmap(jacrev(sc1, argnums=0), in_dims=(None, 0))(f0, xb)  # [b,10,432]
    lead = S.gather(1, yb[:, None]) - S
    gl = G[torch.arange(len(xb)), yb][:, None] - G                # [b,10,432]
    sel = (lead < 2 * v) & (torch.arange(10, device=DEV)[None] != yb[:, None])
    gs = gl[sel]; nsel += len(gs)
    M += gs.T @ gs
print(f"selected (list, competitor) pairs: {nsel}")
ev, V = torch.linalg.eigh(M); V = V[:, ev.argsort(descending=True)]

xt = draw(1_000_000, g); yt = target(xt)
acc = lambda fv: 100 * (predict(unflat(fv, P), xt) == yt).double().mean().item()
print(f"ARC accuracy {acc(f0):.2f}%")
for step in (0.01, 0.1):
    fc = torch.round(f0 / step) * step
    d = f0 - fc
    print(f"--- rounding to {step}: weight change norm {d.norm().item():.3f}")
    for k in (0, 5, 20, 50, 100):
        Vk = V[:, :k]
        print(f"k={k:3d}: accuracy {acc(fc + Vk @ (Vk.T @ d)):.2f}%  (share of change norm restored {((Vk.T@d).norm()/d.norm()).item():.2f})")
    for k in (20, 100):
        r = []
        for _ in range(20):
            Q, _ = torch.linalg.qr(torch.randn(432, k, dtype=DT, device=DEV, generator=g))
            r.append(acc(fc + Q @ (Q.T @ d)))
        print(f"random {k} orthonormal directions (20 draws): mean {np.mean(r):.2f}%, range {min(r):.2f}-{max(r):.2f}")

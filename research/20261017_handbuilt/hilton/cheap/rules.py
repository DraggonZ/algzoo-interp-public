"""Development (no network runs): if each triple's wrong set were 'second-largest below 0, or a below a width A, or b
below a width B', with A and B taken from the reference width functions (widths.py) at one or two nodes, how close
would the estimate be to the converged reduction? Compared per network with ref_triples (converged reduction)."""
import sys, numpy as np, torch
sys.path.insert(0, "/data/output/research/algzoo/20261017_handbuilt/hilton/propagate")
import full_mean
import common as c
log = c.Log("rules.log")
nu, na = 56, 96
V, wt = full_mean.grid(nu, na)
wt = torch.tensor(wt.reshape(nu, na, na), device=c.DEV, dtype=c.dt); mass = wt.sum() * 720
U = torch.tensor(V[:, 2].reshape(nu, na, na), device=c.DEV, dtype=c.dt)
A_ = torch.tensor((V[:, 1] - V[:, 2]).reshape(nu, na, na), device=c.DEV, dtype=c.dt)
B_ = torch.tensor((V[:, 0] - V[:, 1]).reshape(nu, na, na), device=c.DEV, dtype=c.dt)
u = V[:, 2].reshape(nu, na, na)[:, 0, 0]; g = (V[:, 1] - V[:, 2]).reshape(nu, na, na)[0, :, 0]
iu = lambda x: int(np.argmin(np.abs(u - x))); ig = lambda x: int(np.argmin(np.abs(g - x)))


def est(A, B, low=True):
    """A [720, nu, na(b)] width of a-band as function of (u, b); B [720, nu, na(a)] as function of (u, a)."""
    tot = 0.0
    for t in range(720):
        wr = (A_ < A[t][:, None, :]) | (B_ < B[t][:, :, None])
        if low: wr = wr | ((U + A_) < 0)
        tot += (wr.to(c.dt) * wt).sum().item()
    return 100 * (1 - tot / mass.item())


for name in sys.argv[1:] or ["ARC", "200", "208", "212"]:
    d = np.load(f"widths_{name}.npz"); w23 = torch.tensor(d["w23"], device=c.DEV); w12 = torch.tensor(d["w12"], device=c.DEV)
    ref = 100 - 100 * np.load(f"ref_triples_{name}.npz")["err"].sum()
    out = [f"{name:>4s}: converged {ref:.2f}"]
    out.append(f"full widths {est(w23, w12, low=False):.2f}")
    for uu, bb, aa in [(0.84, 0.47, 0.3), (0.66, 0.47, 0.3), (1.02, 0.47, 0.3), (0.84, 0.3, 0.2), (0.84, 0.7, 0.47)]:
        A = w23[:, iu(uu), ig(bb)][:, None, None].expand(720, nu, na); B = w12[:, iu(uu), ig(aa)][:, None, None].expand(720, nu, na)
        out.append(f"one node u={u[iu(uu)]:.2f} b={g[ig(bb)]:.2f} a={g[ig(aa)]:.2f}: {est(A, B):.2f}")
    # two u nodes, linear in u (clipped at >= 0), constant in the other gap
    u1, u2 = iu(0.48), iu(1.2)
    def lin(w, j):
        w1, w2 = w[:, u1, j], w[:, u2, j]; s = torch.tensor((u - u[u1]) / (u[u2] - u[u1]), device=c.DEV)
        return (w1[:, None] + (w2 - w1)[:, None] * s[None]).clamp_min(0)[:, :, None].expand(720, nu, na)
    out.append(f"two u nodes (0.48, 1.20) linear: {est(lin(w23, ig(0.47)), lin(w12, ig(0.3))):.2f}")
    log("; ".join(out))

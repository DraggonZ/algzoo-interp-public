"""Why did propagate/cheap_interp.py (trilinear interpolation of the margin between coarse nodes) converge badly?
For its 10 x 10 x 10 and 20 x 20 x 20 node grids on ARC's network, compare, on its own fine sub-grid, the sign of the
interpolated margin with the sign of the network's margin there (development: the network is run at every fine point).
Mass where they disagree, split by region (second-largest below 0; a < 0.05; b < 0.05; elsewhere) and direction."""
import sys, numpy as np, torch
sys.path.insert(0, "/data/output/research/algzoo/20261017_handbuilt/hilton/propagate")
import cheap_interp as ci
import common as c
log = c.Log("interp_why.log")
W = c.nets()[0][1]
npdf = lambda v: torch.exp(-v * v / 2) / np.sqrt(2 * np.pi); ncdf = lambda v: 0.5 * (1 + torch.erf(v / np.sqrt(2)))
for nu in (10, 20):
    na = nu; sub = 6; R = 4.0; u = np.linspace(-2.0, 3.5, nu); s = np.linspace(0, 1, na); a = R * s ** 2
    U, A, B = np.meshgrid(u, a, a, indexing="ij"); V = np.c_[(U + A + B).ravel(), (U + A).ravel(), U.ravel()]
    Vt = torch.tensor(V, device=c.DEV, dtype=c.dt)
    Mg = ci.margins(W, c.T.repeat_interleave(len(V), 0), Vt.repeat(720, 1)).reshape(720, nu, na, na)
    f = (torch.arange(sub, device=c.DEV, dtype=c.dt) + 0.5) / sub
    uu = torch.tensor(u, device=c.DEV, dtype=c.dt); ss = torch.tensor(s, device=c.DEV, dtype=c.dt)
    # fine points of all cells: coordinates and weights
    ui = (uu[:-1, None] + f[None] * (uu[1:] - uu[:-1])[:, None]).ravel(); du = ((uu[1:] - uu[:-1])[:, None] / sub).expand(-1, sub).ravel()
    si = (ss[:-1, None] + f[None] * (ss[1:] - ss[:-1])[:, None]).ravel(); ds = ((ss[1:] - ss[:-1])[:, None] / sub).expand(-1, sub).ravel()
    ai = R * si ** 2; dai = R * 2 * si * ds
    FU, FA, FB = torch.meshgrid(ui, ai, ai, indexing="ij")
    wt = (npdf(FU + FA + FB) * npdf(FU + FA) * npdf(FU) * ncdf(FU) ** 7 * du[:, None, None] * dai[None, :, None] * dai[None, None, :])
    # interpolation weights: cell index and fraction per axis
    ci_u = torch.arange(nu - 1, device=c.DEV).repeat_interleave(sub); ci_s = torch.arange(na - 1, device=c.DEV).repeat_interleave(sub)
    fu = f.repeat(nu - 1); fs = f.repeat(na - 1)
    cats = {"m2<0": (FU + FA) < 0, "a<0.05": ((FU + FA) >= 0) & (FA < 0.05), "b<0.05": ((FU + FA) >= 0) & (FA >= 0.05) & (FB < 0.05)}
    cats["elsewhere"] = ~(cats["m2<0"] | cats["a<0.05"] | cats["b<0.05"])
    acc = {k: [0.0, 0.0] for k in cats}; tot = wt.sum().item() * 720; est_right = 0.0; true_right = 0.0
    Vf = torch.stack([(FU + FA + FB).ravel(), (FU + FA).ravel(), FU.ravel()], 1)
    for t in range(720):
        Mt = Mg[t]
        m = 0
        for p in (0, 1):
            for q in (0, 1):
                for r in (0, 1):
                    m = m + Mt[ci_u + p][:, None, None].new_zeros(0).sum() if False else m
        # trilinear interpolation, vectorized
        g = Mt  # [nu, na, na]
        iu_, ia_, ib_ = ci_u[:, None, None], ci_s[None, :, None], ci_s[None, None, :]
        x, y, z = fu[:, None, None], fs[None, :, None], fs[None, None, :]
        m = sum(g[iu_ + p, ia_ + q, ib_ + r] * (x if p else 1 - x) * (y if q else 1 - y) * (z if r else 1 - z) for p in (0, 1) for q in (0, 1) for r in (0, 1))
        S, _ = c.scores(W, c.inputs(c.T[t:t + 1].repeat(len(Vf), 1), Vf)); mt, _, _ = c.margin_from(S, c.T[t:t + 1].repeat(len(Vf), 1)[:, 1]); mt = mt.reshape(m.shape)
        ie, it_ = m > 0, mt > 0
        est_right += (ie.to(c.dt) * wt).sum().item(); true_right += (it_.to(c.dt) * wt).sum().item()
        for k, msk in cats.items():
            acc[k][0] += ((ie & ~it_ & msk).to(c.dt) * wt).sum().item(); acc[k][1] += ((~ie & it_ & msk).to(c.dt) * wt).sum().item()
    log(f"nodes {nu}^3 ({720*nu**3:,d} runs): interpolated {100*est_right/tot:.3f}%, network on the same fine points {100*true_right/tot:.3f}%; "
        "interpolation right & network wrong / the reverse, points: " + "; ".join(f"{k} {100*v[0]/tot:.3f}/{100*v[1]/tot:.3f}" for k, v in acc.items()))

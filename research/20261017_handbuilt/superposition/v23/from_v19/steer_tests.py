"""P9: bookkeeping and nudges on v23 from v19, following superposition/steer/lead_parts.py, three_parts.py,
memory_gain.py, balanced.py and phantom.py (same list recipes and seeds)."""
import sys; sys.path.insert(0, "/data/output/research/algzoo/20261017_handbuilt/family")
import torch
from common import DEV, dt, T, ARC, top3, scores
from algzoo.handbuilt_v23_from_v19 import build_v23f

STORE, KEEP, HIDE, FLAG, MEM = [0, 1], [2], [3, 4, 5, 6], [7], list(range(8, 16))
GR = {"store": STORE, "memory": MEM, "hiding": HIDE, "keep gate": KEEP, "flag": FLAG}


def parts(W, Xm, a, b, lm, T_):
    """Exact first-order split of D = score(a) - score(b) at the end of step T_, by group and source input."""
    Xm = Xm.clone().requires_grad_(True); h = torch.zeros(len(Xm), 16, device=DEV, dtype=dt)
    for t in range(T_ + 1): h = torch.relu(h @ W[1].T + Xm[:, t:t + 1] * W[0])
    ht = h.detach().requires_grad_(True); h2 = ht
    for t in range(T_ + 1, 10): h2 = torch.relu(h2 @ W[1].T + Xm[:, t:t + 1].detach() * W[0])
    sc = h2 @ W[2].T; n = torch.arange(len(Xm), device=DEV); D = sc[n, a] - sc[n, b]
    gD, = torch.autograd.grad(D.sum(), ht)
    Jt = torch.stack([torch.autograd.grad(h[:, k].sum(), Xm, retain_graph=True)[0] for k in range(16)], 1)
    C = gD[:, :, None] * Jt * Xm.detach()[:, None, :]
    out = {}
    for gname, g in GR.items():
        Cg = C[:, g].sum(1); out[gname] = (Cg[n, a], Cg[n, lm], Cg.sum(1) - Cg[n, a] - Cg[n, lm])
    return out, D.detach()


def run(W, X, T_=None, scale=(), extra=0, x_extra=0.0):
    h = torch.zeros(len(X), 16, device=DEV, dtype=dt)
    for t in range(10):
        h = torch.relu(h @ W[1].T + X[:, t:t + 1] * W[0])
        if t == T_:
            h = h.clone()
            for g, f in scale: h[:, g] *= f
    for _ in range(extra): h = torch.relu(h @ W[1].T + x_extra * W[0])
    return (h @ W[2].T).argmax(1)


def report(W, lab):
    print(f"===== {lab}")
    g = torch.Generator(device=DEV).manual_seed(5); X = torch.randn(8_000_000, 10, device=DEV, dtype=dt, generator=g)
    v = X.sort(1, descending=True).values; l, s, t3 = top3(X)
    near = (v[:, 1] > 0) & ((v[:, 1] - v[:, 2]) < 0.05 * v[:, 1]); X, l, s, t3 = X[near], l[near], s[near], t3[near]
    T_ = 5; bal = {}
    for who, A_, B_ in (("second early, third late", s, t3), ("third early, second late", t3, s)):
        m = (l <= T_ - 3) & (A_ >= l + 2) & (A_ <= T_ - 2) & (B_ >= T_ + 1); Xm, am, bm, lm = X[m][:20000], A_[m][:20000], B_[m][:20000], l[m][:20000]
        out, D = parts(W, Xm, am, bm, lm, T_)
        print(f"(a) step {T_}, {who}: final lead median {D.median():.3f}; median part at step {T_} from the earlier contender / the largest / the rest")
        for gname, (pa, pl, pr) in out.items():
            print(f"   {gname:10s} {pa.median():+9.4f} {pl.median():+9.4f} {pr.median():+9.4f}")
        bal[who] = (out["memory"][1].median() / out["store"][1].median()).abs().item()
        tot_e = sum(out[k][0] for k in out).median().item(); dm = (out["memory"][0].median() + 0 * tot_e).item()
        bal[who + " conv"] = (out["memory"][0].median() + bal[who] * out["store"][0].median()).item() / tot_e
    b = sum(bal[k] for k in bal if "conv" not in k) / 2; conv = sum(bal[k] for k in bal if "conv" in k) / 2
    print(f"   balance b = |memory's part from the largest / store's| = {b:.3f}; value-scale conversion {conv:.3f}")
    m = (l <= 3); Xm, lm, sm = X[m][:20000], l[m][:20000], s[m][:20000]
    out, D = parts(W, Xm, sm, lm, lm, T_)
    print(f"(b) lead of the second over the largest, median {D.median():.3f}; per group at step 5, median part from: the second / the largest / the rest")
    for gname, (pa, pl, pr) in out.items(): print(f"   {gname:10s} {pa.median():+9.4f} {pl.median():+9.4f} {pr.median():+9.4f}")
    for who, A_, B_ in (("second early, third late", s, t3), ("third early, second late", t3, s)):
        m = (l <= T_ - 3) & (A_ >= l + 2) & (A_ <= T_ - 2) & (B_ >= T_ + 1); Xm, am, lm = X[m], A_[m], l[m]
        pa = lambda p: f"{100*(p == am).double().mean().item():5.1f}/{100*(p == lm).double().mean().item():4.1f}"
        print(f"(c, d) step {T_}, {who} (n {len(Xm):,}); earlier contender wins / largest wins: base {pa(run(W, Xm))}")
        print(f"   memory alone x0.95 {pa(run(W, Xm, T_, [(MEM, 0.95)]))} | x1.05 {pa(run(W, Xm, T_, [(MEM, 1.05)]))}")
        for a in (-0.04, -0.02, 0.02, 0.04):
            Xc = Xm.clone(); Xc[torch.arange(len(Xm), device=DEV), am] *= (1 + conv * a)
            print(f"   a {a:+.2f}: balanced memory x{1+a:.2f}, store x{1+b*a:.4f}: {pa(run(W, Xm, T_, [(MEM, 1 + a), (STORE, 1 + b * a)]))} | predicted (value x {1+conv*a:.3f}) {pa(run(W, Xc))}", flush=True)
    g = torch.Generator(device=DEV).manual_seed(71); X = torch.randn(1_000_000, 10, device=DEV, dtype=dt, generator=g); o = X.argsort(1, descending=True); L, S = o[:, 0], o[:, 1]
    for k in (1, 2):
        a = run(W, X, extra=k); m = S >= k
        print(f"(e) {k} extra step(s), input 0: names second-{k} {100*(a[m] == S[m] - k).double().mean().item():5.1f}% | the second {100*(a[m] == S[m]).double().mean().item():5.1f}% (own accuracy {100*(run(W, X) == S).double().mean().item():.1f}%)", flush=True)


if __name__ == "__main__":
    report(T(build_v23f().params), "v23 from v19, loop gain 1e-4")
    report(T(build_v23f(ep=0.3).params), "v23 from v19, loop gain 0.3 (ARC-sized ring)")

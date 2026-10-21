"""Leanings at a tie, fitted locally per arrangement. Third against second: lists with the second positive and the
third within 10% of it; per arrangement, D_T / x_S = lean + slope * (1 - x_T / x_S) + k * (x_L / x_S - 1.5), in votes.
Largest against second (overtaking near ties): the largest within 10% of the second, second positive;
D_L / x_S = lean + slope * (x_L / x_S - 1) + k * (x_T / x_S - 0.8). lean > 0 favours the wrong answer. ARC and v7z;
also the spread of D around each cell's local fit (the leftover's size near ties, in votes). Saved for later steps."""
import sys, numpy as np, torch
sys.path.insert(0, "../../family"); sys.path.insert(0, "..")
from common import *
from config_rule import cells
from typical import vote_unit, cell_info, ORD
def local(W, kind, n=12000000, seed=41, f=None):
    g = torch.Generator(device=DEV).manual_seed(seed); u = vote_unit(W, torch.randn(20000, 10, device=DEV, dtype=dt, generator=g)) if f is None else 1.0
    A = torch.zeros(1000, 4, 4, device=DEV, dtype=dt); B = torch.zeros(1000, 4, device=DEV, dtype=dt); YY = torch.zeros(1000, device=DEV, dtype=dt); N = torch.zeros(1000, device=DEV, dtype=dt)
    for _ in range(n // 1000000):
        X = torch.randn(1000000, 10, device=DEV, dtype=dt, generator=g); o = X.argsort(1, descending=True); ar = torch.arange(len(X), device=DEV)
        xl, xs, xt = X[ar, o[:, 0]], X[ar, o[:, 1]], X[ar, o[:, 2]]
        m = (xs > 0) & ((1 - xt / xs) < 0.1) if kind == "T" else (xs > 0) & ((xl / xs - 1) < 0.1)
        X, o, xl, xs, xt = X[m], o[m], xl[m], xs[m], xt[m]; ar = torch.arange(len(X), device=DEV)
        Sc = (scores(W, X) if f is None else f(X)) / u
        if kind == "T": y = (Sc[ar, o[:, 2]] - Sc[ar, o[:, 1]]) / xs; F = torch.stack([torch.ones_like(xs), 1 - xt / xs, xl / xs - 1.5], 1)
        else: y = (Sc[ar, o[:, 0]] - Sc[ar, o[:, 1]]) / xs; F = torch.stack([torch.ones_like(xs), xl / xs - 1, xt / xs - 0.8], 1)
        F = torch.cat([F, torch.zeros_like(xs)[:, None]], 1); c = cells(X, "exact arrangement")
        A.index_add_(0, c, F[:, :, None] * F[:, None, :]); B.index_add_(0, c, F * y[:, None]); YY.index_add_(0, c, y * y); N.index_add_(0, c, torch.ones_like(y))
    A[:, 3, 3] += 1.0
    sol = torch.linalg.solve(A + 1e-9 * torch.eye(4, device=DEV, dtype=dt), B[..., None])[..., 0]
    rss = YY - 2 * (sol * B).sum(1) + torch.einsum("ci,cij,cj->c", sol, A, sol) - sol[:, 3] ** 2
    return sol[:, :3].cpu().numpy(), np.sqrt((rss / N.clamp(min=1)).clamp(min=0).cpu().numpy()), N.cpu().numpy()
if __name__ == "__main__":
    from baseline_v7z import nets
    L, S, Tq, valid, order = cell_info(); res = {}
    for nm, W in (("ARC", ARC), ("v7z", nets()["v7 + z + ninth slot (18)"])):
        for kind in ("T", "L"): res[(nm, kind)] = local(W, kind)
    np.savez("local_lean.npz", **{f"{a}_{b}_{i}": v for (a, b), t in res.items() for i, v in enumerate(t)})
    for kind, title, groups in (("T", "third against second", [("all", valid)] + [(f"order {o}", valid & (order == o)) for o in ORD] + [(f"third at {p}", valid & (Tq == p)) for p in range(10)]),
                                ("L", "largest against second (overtaking near ties)", [("S before L", valid & (S < L))] + [(f"largest at {p}, S before", valid & (S < L) & (L == p)) for p in range(1, 10)])):
        print(f"\n== {title}: median over arrangements of lean (votes, >0 favours the wrong answer), slope, and the spread around the fit")
        print(f"{'group':26s} {'ARC lean':>9s} {'v7z lean':>9s} {'ARC slope':>10s} {'v7z slope':>10s} {'ARC spread':>11s} {'v7z spread':>11s}")
        for nm, m in groups:
            m = m & (res[("ARC", kind)][2] > 50)
            a, b = res[("ARC", kind)], res[("v7z", kind)]
            print(f"{nm:26s} {np.median(a[0][m, 0]):+9.3f} {np.median(b[0][m, 0]):+9.3f} {np.median(a[0][m, 1]):+10.2f} {np.median(b[0][m, 1]):+10.2f} {np.median(a[1][m]):11.3f} {np.median(b[1][m]):11.3f}")

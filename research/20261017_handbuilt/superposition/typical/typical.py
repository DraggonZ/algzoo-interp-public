"""Typical rules per arrangement, for any network: for each exact arrangement of the three largest (positions of L, S, T)
a linear rule, fitted by least squares on random lists, for D_T = score_T - score_S and D_L = score_L - score_S as
a x_L + b x_S + c x_T, in votes (scores divided by the network's vote per unit). Leanings at a tie: lean_T = D_T / x_S at
x_T = x_S and x_L = rho x_S (rho = the cell's median ratio), i.e. how much the third is favoured over an equal second;
lean_L = D_L / x_S at x_L = x_S (how much an overtaking largest is favoured over an equal second), x_T at its median.
Negative leanings favour the right answer. Saved per network for later steps."""
import sys, numpy as np, torch
sys.path.insert(0, "../../family"); sys.path.insert(0, "..")
from common import *
from config_rule import cells
def vote_unit(W, X):
    J = jac(W, X[:20000]); L = X[:20000].argmax(1)
    kept = (X[:20000] > 0) & (torch.arange(10, device=DEV)[None] != L[:, None]); return J[:, torch.arange(10), torch.arange(10)][kept].median()
def typical(W, n=4000000, seed=31, f=None):
    g = torch.Generator(device=DEV).manual_seed(seed); A = torch.zeros(1000, 3, 3, device=DEV, dtype=dt); B = torch.zeros(1000, 3, 2, device=DEV, dtype=dt)
    u = vote_unit(W, torch.randn(20000, 10, device=DEV, dtype=dt, generator=g)) if f is None else 1.0
    med = torch.zeros(1000, 3, device=DEV, dtype=dt); cnt = torch.zeros(1000, device=DEV, dtype=dt)
    for _ in range(n // 500000):
        X = torch.randn(500000, 10, device=DEV, dtype=dt, generator=g); o = X.argsort(1, descending=True); ar = torch.arange(len(X), device=DEV)
        v = torch.stack([X[ar, o[:, k]] for k in range(3)], 1); Sc = (scores(W, X) if f is None else f(X)) / u
        y = torch.stack([Sc[ar, o[:, 2]] - Sc[ar, o[:, 1]], Sc[ar, o[:, 0]] - Sc[ar, o[:, 1]]], 1); c = cells(X, "exact arrangement")
        A.index_add_(0, c, v[:, :, None] * v[:, None, :]); B.index_add_(0, c, v[:, :, None] * y[:, None, :])
        med.index_add_(0, c, v / v[:, 1:2].abs().clamp(min=1e-3)); cnt.index_add_(0, c, torch.ones(len(X), device=DEV, dtype=dt))
    sol = torch.linalg.solve(A + 1e-6 * torch.eye(3, device=DEV, dtype=dt), B)   # [cell, (L,S,T), (D_T, D_L)]
    return sol, med / cnt.clamp(min=1)[:, None]
def leanings(sol, mean_ratio):
    rho = mean_ratio[:, 0]; tau = mean_ratio[:, 2]
    lean_T = sol[:, 0, 0] * rho + sol[:, 1, 0] + sol[:, 2, 0]
    lean_L = sol[:, 0, 1] + sol[:, 1, 1] + sol[:, 2, 1] * tau
    return lean_T, lean_L
ORD = {"LST": (0, 1, 2), "LTS": (0, 2, 1), "SLT": (1, 0, 2), "TLS": (1, 2, 0), "STL": (2, 0, 1), "TSL": (2, 1, 0)}
def cell_info():
    L, S, Tq = np.divmod(np.arange(1000), 100)[0], np.divmod(np.arange(1000), 10)[0] % 10, np.arange(1000) % 10
    valid = (L != S) & (L != Tq) & (S != Tq); order = np.array(["".join(sorted("LST", key=lambda ch: {"L": l, "S": s, "T": t}[ch])) for l, s, t in zip(L, S, Tq)])
    return L, S, Tq, valid, order
if __name__ == "__main__":
    from baseline_v7z import nets
    out = {}
    for nm, W in (("ARC", ARC), ("v7z", nets()["v7 + z + ninth slot (18)"])):
        sol, mr = typical(W); lt, ll = leanings(sol, mr); out[nm] = (sol.cpu().numpy(), mr.cpu().numpy(), lt.cpu().numpy(), ll.cpu().numpy())
    np.savez("typical_ARC_v7z.npz", **{f"{k}_{i}": v for k, t in out.items() for i, v in enumerate(t)})
    L, S, Tq, valid, order = cell_info()
    print("Leaning toward the third at a tie (votes per unit of the second; negative favours the right answer), median over cells:")
    print(f"{'group':28s} {'ARC':>8s} {'v7z':>8s} {'ARC spread (10-90%)':>22s}")
    def row(nm, m):
        a, b = out["ARC"][2][m], out["v7z"][2][m]; print(f"{nm:28s} {np.median(a):+8.3f} {np.median(b):+8.3f}   [{np.percentile(a,10):+.2f}, {np.percentile(a,90):+.2f}]")
    row("all", valid)
    for o in ORD: row(f"order {o}", valid & (order == o))
    for p in range(10): row(f"third at position {p}", valid & (Tq == p))
    print("\nLeaning toward the largest at a tie with the second (overtakings: S before L):")
    def rowL(nm, m):
        a, b = out["ARC"][3][m], out["v7z"][3][m]; print(f"{nm:28s} {np.median(a):+8.3f} {np.median(b):+8.3f}   [{np.percentile(a,10):+.2f}, {np.percentile(a,90):+.2f}]")
    rowL("all, S before L", valid & (S < L))
    for p in range(1, 10): rowL(f"largest at {p}, S before", valid & (S < L) & (L == p))
    rowL("all, S after L", valid & (S > L))

"""The record waves in the scores, ARC against v7 + z + ninth slot. For each group (memory, hiding/clock, all), the exact
per-list weight of the largest L and of the previous largest P on each position's final score, centred over positions
0-9 (adding the same to every score changes nothing), median over random lists, by the position of L. Rows: position of
L; columns: offset of the scored position from L (-9..9). Divided by the network's vote per unit (median own weight of
a kept number) so both networks are in votes."""
import sys, numpy as np, torch
sys.path.insert(0, "../../family"); sys.path.insert(0, "..")
from common import *
from baseline_v7z import nets
def group_jac(W, X, idx):
    X = X.clone().requires_grad_(True); h = torch.zeros(len(X), len(W[0]), device=DEV, dtype=dt)
    for t in range(10): h = torch.relu(h @ W[1].T + X[:, t:t + 1] * W[0])
    S = h[:, idx] @ W[2][:, idx].T; S = S - S.mean(1, keepdim=True); J = []
    for p in range(10):
        g, = torch.autograd.grad(S[:, p].sum(), X, retain_graph=p < 9); J.append(g)
    return torch.stack(J, 1).detach()          # [n, p, s]
def show(name, W, groups, X):
    n = len(X); ar = torch.arange(n, device=DEV); m = torch.zeros(n, device=DEV, dtype=dt); Lp = torch.full((n,), -1, device=DEV, dtype=torch.long); Pp = Lp.clone()
    for t in range(10):
        new = X[:, t] > m; Pp = torch.where(new, Lp, Pp); Lp = torch.where(new, t, Lp); m = torch.where(new, X[:, t], m)
    Jall = group_jac(W, X, list(range(len(W[0])))); kept = (X > 0) & (torch.arange(10, device=DEV)[None] != Lp[:, None])
    own = Jall[:, torch.arange(10), torch.arange(10)]; vote = own[kept].median()
    print(f"==== {name}: vote per unit {vote:.4g}")
    for gname, idx in groups.items():
        J = group_jac(W, X, idx) / vote
        for role, Pos in (("L", Lp), ("P", Pp)):
            print(f"  {gname}, weight of {role} on the score at offset (position - L's position), median, rows = L's position 1..9:")
            for r in range(1, 10):
                mm = (Lp == r) & (Pos >= 0)
                if mm.sum() < 200: continue
                wv = J[mm][:, :, Pos[mm][0]] if False else J[mm, :, :][torch.arange(mm.sum().item(), device=DEV), :, Pos[mm]]
                med = wv.median(0).values.cpu().numpy()
                print(f"    L at {r}: " + " ".join(f"{('%+.2f' % med[p]) if True else '':>6s}" for p in range(10)))
if __name__ == "__main__":
    X = torch.randn(150000, 10, device=DEV, dtype=dt, generator=torch.Generator(device=DEV).manual_seed(3))
    show("ARC", ARC, {"memory (0,3,10-15)": [0, 3, 10, 11, 12, 13, 14, 15], "hiding (1,6,7,8)": [1, 6, 7, 8], "all neurons": list(range(16))}, X)
    W = nets()["v7 + z + ninth slot (18)"]
    show("v7z", W, {"memory (9 slots, 8-16)": list(range(8, 17)), "clock, second, z (4-7, 17)": [4, 5, 6, 7, 17], "all neurons": list(range(len(W[0])))}, X)

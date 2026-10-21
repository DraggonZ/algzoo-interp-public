"""v22 step 1: imperfect hiding. ARC's hidden largest leaves a leftover on every score (waves.log, 'all neurons, weight
of L'): about +-0.1-0.2 votes per unit of the largest, patterned by where the largest sits. v7z hides exactly. Rule: the
readout's subtraction of the clock's 2-D phase is slightly off per position: add V_p . phase to score p (V: 10 x 2, 20
numbers), with V the least-squares match of ARC's measured leftover table (largest at 1..8; analysis fit on ARC's
measured behaviour, not on mistakes). The clock's phase per unit of the largest, by the largest's position, is read
from v7z's own runs. Scored by accuracy, ARC's mistakes reproduced, near ties, and the role map both ways."""
import sys, numpy as np, torch
sys.path.insert(0, "../../family"); sys.path.insert(0, "..")
from common import *
import score
from baseline_v7z import nets, vz
from waves import group_jac
from algzoo.handbuilt import TO_PQ
CLK = [4, 5, 6]
def tracks(X):
    n = len(X); m = torch.zeros(n, device=DEV, dtype=dt); Lp = torch.full((n,), -1, device=DEV, dtype=torch.long)
    for t in range(10):
        new = X[:, t] > m; Lp = torch.where(new, t, Lp); m = torch.where(new, X[:, t], m)
    return Lp
def leftover_table(W, X):
    """median weight of L on each centred final score, all neurons, in votes, by L's position (rows 0-9)."""
    Lp = tracks(X); J = group_jac(W, X, list(range(len(W[0])))); ar = torch.arange(len(X), device=DEV)
    kept = (X > 0) & (torch.arange(10, device=DEV)[None] != Lp[:, None]); own = J[:, torch.arange(10), torch.arange(10)]; vote = own[kept].median()
    Tb = torch.zeros(10, 10, device=DEV, dtype=dt)
    for r in range(10):
        m = Lp == r
        if m.sum() > 100: Tb[r] = (J[m][torch.arange(m.sum().item(), device=DEV), :, r] / vote).median(0).values
    return Tb, vote
def phase_by_pos(W, X):
    """v7z's final clock phase (2-D, TO_PQ coordinates) per unit of the largest, median by L's position."""
    Lp = tracks(X); h = torch.zeros(len(X), len(W[0]), device=DEV, dtype=dt)
    for t in range(10): h = torch.relu(h @ W[1].T + X[:, t:t + 1] * W[0])
    ph = h[:, CLK] @ torch.tensor(TO_PQ, device=DEV, dtype=dt).T; xl = X.max(1).values
    return torch.stack([(ph[Lp == r] / xl[Lp == r, None]).median(0).values if (Lp == r).sum() > 100 else torch.zeros(2, device=DEV, dtype=dt) for r in range(10)])
def build(scale=1.0):
    W = nets()["v7 + z + ninth slot (18)"]
    X = torch.randn(150000, 10, device=DEV, dtype=dt, generator=torch.Generator(device=DEV).manual_seed(3))
    Ta, _ = leftover_table(ARC, X); _, vz_vote = leftover_table(W, X); Ph = phase_by_pos(W, X)
    rows = list(range(1, 9)); A = Ph[rows]                 # [8, 2]
    Tc = Ta[rows] - Ta[rows].mean(1, keepdim=True)          # target leftover, centred over positions, in votes
    V = torch.linalg.lstsq(A, Tc).solution.T                # [10, 2]: score_p += V_p . phase (in votes per unit)
    fit = A @ V.T; r2 = 1 - ((fit - Tc) ** 2).sum() / (Tc ** 2).sum()
    wo = W[2].clone(); wo[:, CLK] += scale * vz_vote * (V @ torch.tensor(TO_PQ, device=DEV, dtype=dt))
    return [W[0], W[1], wo], r2.item(), V
if __name__ == "__main__":
    models = {}
    for s in (1.0,):
        Wn, r2, V = build(s); print(f"leftover table matched by the 20-number phase readout: {100*r2:.0f}% of its (centred) sum of squares")
        models[f"v7z + hiding leftover (x{s})"] = [Wn]
    models["v7z"] = [nets()["v7 + z + ninth slot (18)"]]
    score.report(models)

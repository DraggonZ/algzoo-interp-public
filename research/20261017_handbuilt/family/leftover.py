"""A short joint explanation of ARC's scores: ideal votes plus the hiding leftover.
score_p (centred over positions) ~ A_p x_p [x_p > 0, p not the largest]          (votes; 10 numbers)
                                 + c[r, p] x_L + d[r, p] x_P                       (leftover of hiding the largest at r;
                                                                                   P = previous largest; 200 numbers)
Least squares on 400,000 random lists (analysis fit on the trained network). Ladder: votes only; + leftover on the
largest; + on the previous largest; + by sign of the second (the network keeps only positive numbers, so the second's
vote is replaced by A_p x_p+). Each model is scored as a stand-alone scorer by ARC's mistakes reproduced and near-tie
behaviour (score.py)."""
import numpy as np, torch
from common import *
import score
def feats(X, parts):
    n = len(X); ar = torch.arange(n, device=DEV); o = X.argsort(1, descending=True)
    # largest and previous largest (the record the largest beat), positions
    m = torch.zeros(n, device=DEV, dtype=dt); Lp = torch.full((n,), -1, device=DEV, dtype=torch.long); Pp = Lp.clone()
    for t in range(10):
        new = X[:, t] > m; Pp = torch.where(new, Lp, Pp); Lp = torch.where(new, t, Lp); m = torch.where(new, X[:, t], m)
    xL = torch.where(Lp >= 0, X[ar, Lp.clamp(min=0)], 0); xP = torch.where(Pp >= 0, X[ar, Pp.clamp(min=0)], 0)
    F = []
    eye = torch.eye(10, device=DEV, dtype=dt)
    keep = (X > 0) & (torch.arange(10, device=DEV)[None] != Lp[:, None])
    F.append(eye[None] * (X * keep)[:, :, None])                     # [n, p, 10]: A_p x_p+
    if "L" in parts:
        oh = torch.zeros(n, 10, 100, device=DEV, dtype=dt); idx = (Lp.clamp(min=0) * 10)[:, None] + torch.arange(10, device=DEV)[None]
        oh.scatter_(2, idx[:, :, None], (xL * (Lp >= 0))[:, None, None].expand(n, 10, 1)); F.append(oh)
    if "P" in parts:
        oh = torch.zeros(n, 10, 100, device=DEV, dtype=dt); idx = (Lp.clamp(min=0) * 10)[:, None] + torch.arange(10, device=DEV)[None]
        oh.scatter_(2, idx[:, :, None], (xP * (Lp >= 0))[:, None, None].expand(n, 10, 1)); F.append(oh)
    return torch.cat(F, 2)
def fit(parts, n=400000, lam=1e-3):
    """Fit on the positions of the three largest only, centred over those three (what decides the answer)."""
    X = torch.randn(n, 10, device=DEV, dtype=dt, generator=torch.Generator(device=DEV).manual_seed(21))
    ar = torch.arange(n, device=DEV)[:, None]; top = X.argsort(1, descending=True)[:, :3]
    Y = scores(ARC, X)[ar, top]; Y = Y - Y.mean(1, keepdim=True)
    F = feats(X, parts)[ar, top]; F = F - F.mean(1, keepdim=True)
    A = F.reshape(-1, F.shape[2]); b = Y.reshape(-1)
    G = A.T @ A; w = torch.linalg.solve(G + lam * torch.trace(G) / len(G) * torch.eye(len(G), device=DEV, dtype=dt), A.T @ b)
    r2 = 1 - ((A @ w - b) ** 2).mean() / b.var()
    return w, r2.item()
def model(parts):
    w, r2 = fit(parts); print(f"  fit {parts or 'votes only'}: {len(w)} numbers, R2 of centred scores {r2:.3f}; votes per unit by position {np.round(w[:10].cpu().numpy(), 1)}")
    return (lambda X: torch.cat([feats(X[i:i + 100000], parts) @ w for i in range(0, len(X), 100000)])), w
if __name__ == "__main__":
    models = {}; W = {}
    for nm, parts in (("votes only", ""), ("votes + leftover on L", "L"), ("votes + leftover on L and P", "LP")):
        f, w = model(parts); models[nm] = [f]; W[nm] = w.cpu().numpy()
    np.savez("leftover_fit.npz", **{k.replace(" ", "_"): v for k, v in W.items()})
    models["ARC itself"] = [ARC]
    score.report(models)

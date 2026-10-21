"""How many weight directions must be exact? Sensitivity: M = sum over lists and competitors j of 1[lead_j < 2] g_j g_j^T,
g_j = gradient of the lead (right answer's score minus j's) with respect to all 432 weights, on 40,000 random lists
(the near-tie directions of ../../hilton/hilton_lowrank.py). Clean weights: ARC's weights rounded to a grid (0.01,
0.05, 0.1) or with weights below 0.1 in size set to zero. Restored: W_k = W_clean + V_k V_k^T (W_ARC - W_clean), V_k the top
k eigenvectors of M. For k = 0 ... 432: accuracy and ARC's mistakes reproduced (grid) and agreement on random lists.
Control: restoring k random directions."""
import sys, numpy as np, torch
from torch.func import vmap, grad
sys.path.insert(0, "../../family")
from common import *
import score
wi0, wh0, wo0 = ARC; sizes = [wi0.numel(), wh0.numel(), wo0.numel()]
w0 = torch.cat([wi0.flatten(), wh0.flatten(), wo0.flatten()])
def unflat(v):
    a, b = sizes[0], sizes[0] + sizes[1]; return [v[:a], v[a:b].reshape(16, 16), v[b:].reshape(10, 16)]
def logits(v, x):
    wi, wh, wo = unflat(v); h = torch.zeros(16, device=DEV, dtype=dt)
    for t in range(10): h = torch.relu(wh @ h + x[t] * wi)
    return wo @ h
if __name__ == "__main__":
    g = torch.Generator(device=DEV).manual_seed(61); X = torch.randn(40000, 10, device=DEV, dtype=dt, generator=g); y = X.argsort(1, descending=True)[:, 1]
    M = torch.zeros(432, 432, device=DEV, dtype=dt)
    for i in range(0, len(X), 2000):
        xb, yb = X[i:i + 2000], y[i:i + 2000]
        Jl = vmap(torch.func.jacrev(logits), in_dims=(None, 0))(w0, xb)          # [b, 10, 432]
        S = vmap(logits, in_dims=(None, 0))(w0, xb)
        ar = torch.arange(len(xb), device=DEV)
        lead = S[ar, yb][:, None] - S; gl = Jl[ar, yb][:, None, :] - Jl          # [b, 10, 432]
        m = (lead < 2) & (torch.arange(10, device=DEV)[None] != yb[:, None])
        G = gl[m]; M += G.T @ G
    ev, V = torch.linalg.eigh(M); V = V.flip(1); ev = ev.flip(0)
    print("share of sensitivity in the top k directions: " + " ".join(f"k={k}: {(ev[:k].sum()/ev.sum()).item():.3f}" for k in (1, 5, 10, 20, 50, 100)))
    cleans = {"rounded to 0.01": torch.round(w0 / 0.01) * 0.01, "rounded to 0.05": torch.round(w0 / 0.05) * 0.05, "rounded to 0.1": torch.round(w0 / 0.1) * 0.1,
              "weights under 0.1 set to zero": torch.where(w0.abs() < 0.1, 0.0, w0)}
    Q, _ = torch.linalg.qr(torch.randn(432, 432, device=DEV, dtype=dt, generator=torch.Generator(device=DEV).manual_seed(3)))
    models = {}
    for nm, wc in cleans.items():
        d = w0 - wc; print(f"{nm}: size of the change RMS {d.pow(2).mean().sqrt():.4f}; its share in the top 20 sensitive directions {((V[:, :20].T @ d).pow(2).sum() / d.pow(2).sum()).item():.3f}")
        for k in (0, 5, 10, 20, 50, 100):
            Vk = V[:, :k]; models[f"{nm}, top {k} restored"] = [unflat(wc + Vk @ (Vk.T @ d))]
        Qk = Q[:, :20]; models[f"{nm}, 20 random restored"] = [unflat(wc + Qk @ (Qk.T @ d))]
    score.report(models, family=False)

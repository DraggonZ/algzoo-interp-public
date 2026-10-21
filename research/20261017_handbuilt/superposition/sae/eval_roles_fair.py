"""Fairer role matching: before correlating, remove from every unit (roles, ARC neurons, SAE latents) the part
explained linearly by the shared quantities the largest so far, the current number and the step (one-hot), pooled over
steps. Then best |correlation| per role and roles above 0.9 / 0.7, as eval_sae.py."""
import sys, numpy as np, torch
sys.path.insert(0, "../../family"); sys.path.insert(0, "..")
from common import *
from train_sae import states
from eval_sae import load, sd
from role_map import states as rstates
from baseline_v7z import nets
from algzoo.handbuilt_v19 import build_v19
from algzoo.handbuilt_v21 import build_v21
def shared(X):
    m = torch.zeros(len(X), device=DEV, dtype=dt); F = []
    for t in range(10):
        m = torch.maximum(m, X[:, t]); F.append(torch.cat([m[:, None], X[:, t:t + 1], torch.nn.functional.one_hot(torch.full((len(X),), t, device=DEV), 10).to(dt)], 1))
    return torch.stack(F, 1).reshape(-1, 12)
def resid(U, F):
    W = torch.linalg.lstsq(F, U).solution; return U - F @ W
if __name__ == "__main__":
    g = torch.Generator(device=DEV).manual_seed(78); X = torch.randn(60000, 10, device=DEV, dtype=dt, generator=g); F = shared(X)
    H = states(X).reshape(-1, 16)
    roles = {"v19": rstates(T(build_v19(kappa=0.5).params), X)[0].reshape(-1, 16), "v21": rstates(T(build_v21()), X)[0].reshape(-1, 28),
             "v7z": rstates(nets()["v7 + z + ninth slot (18)"], X)[0].reshape(-1, 18)}
    def best(R, Z):
        R = resid(R[:, R.std(0) > 1e-9], F); Z = resid(Z[:, Z.std(0) > 1e-9], F)
        a = (R - R.mean(0)) / (R.std(0) + 1e-12); b = (Z - Z.mean(0)) / (Z.std(0) + 1e-12); C = (a.T @ b / len(a)).abs(); bm = C.max(1).values
        keep = R.std(0) > 1e-6 * R.std(0).max()
        bm = bm[keep]; return f"best {bm.mean():.2f}, above 0.9 {(bm > 0.9).sum().item()}, above 0.7 {(bm > 0.7).sum().item()} of {len(bm)}"
    dicts = {"ARC's 16 neurons": H}
    for m, k in [(16, 4), (32, 4), (128, 4), (128, 8)]:
        with torch.no_grad(): dicts[f"SAE {m}x k{k}"] = load(m, k).encode(H.float() / sd).to(dt)
    for nm, Z in dicts.items():
        print(f"{nm:18s} | " + " | ".join(f"{rn}: {best(R, Z)}" for rn, R in roles.items()), flush=True)

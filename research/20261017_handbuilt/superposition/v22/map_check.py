"""Criterion 1 for the v22 steps so far: ARC's neurons from the candidate's roles and the roles from ARC's neurons
(held-out R^2, one linear map shared over steps, plus the current number), as baseline_v7z.py."""
import sys, numpy as np, torch
sys.path.insert(0, "../../family"); sys.path.insert(0, "..")
from common import *
from role_map import states
from reverse_map import ridge_r2
import hide_leftover as hl
from reveal_spill import v7z
if __name__ == "__main__":
    g = torch.Generator(device=DEV).manual_seed(77); X = torch.randn(200000, 10, device=DEV, dtype=dt, generator=g)
    tr, te = slice(0, 100000), slice(100000, None); Ha = states(ARC, X)[0]
    fl = lambda A, s: torch.cat([A[s], X[s][:, :, None]], 2).reshape(-1, A.shape[2] + 1)
    C = {"v7z": v7z(0.8)}
    hl.nets = lambda: {"v7 + z + ninth slot (18)": v7z(0.8)}; C["v7z + hiding leftover"] = hl.build()[0]
    hl.nets = lambda: {"v7 + z + ninth slot (18)": v7z(0.6)}; C["v7z w 0.6 + hiding leftover"] = hl.build()[0]
    for nm, W in C.items():
        R = states(W, X)[0]; R = R / (R[tr].reshape(-1, R.shape[2]).std(0) + 1e-12); Hn = Ha / (Ha[tr].reshape(-1, 16).std(0) + 1e-12)
        f = ridge_r2(fl(R, tr), fl(R, te), Ha[tr].reshape(-1, 16), Ha[te].reshape(-1, 16))
        r = ridge_r2(fl(Hn, tr), fl(Hn, te), R[tr].reshape(-1, R.shape[2]), R[te].reshape(-1, R.shape[2]))
        print(f"{nm:28s} ARC from roles {f.mean():.3f} [" + " ".join(f"{v:.2f}" for v in f.tolist()) + f"] | roles from ARC {r.mean():.3f}")

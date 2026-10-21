"""Role comparison as ../sae/eval_roles_fair.py, with transcoder latents: on 60,000 random lists (same seed, 78), remove
from every unit the part explained linearly by the largest so far, the current number and the step (one-hot), then for
each hand-built role (v19, v21, v7z) the best |correlation| with any unit. Transcoder latents are computed from ARC's
real (h_{t-1}, x_t). Rows for ARC's neurons and the SAEs are recomputed here for a side-by-side check."""
import sys, os, torch
sys.path.insert(0, "../../family"); sys.path.insert(0, ".."); sys.path.insert(0, "../sae")
from common import *
from train_tc import pairs, states
from eval_tc import load as load_tc, sd as sd_tc
from role_map import states as rstates
from baseline_v7z import nets
from algzoo.handbuilt_v19 import build_v19
from algzoo.handbuilt_v21 import build_v21
from train_sae import SAE
here = os.getcwd(); os.chdir("../sae"); import eval_roles_fair as erf; from eval_sae import load as load_sae, sd as sd_sae; os.chdir(here)
if __name__ == "__main__":
    g = torch.Generator(device=DEV).manual_seed(78); X = torch.randn(60000, 10, device=DEV, dtype=dt, generator=g); F = erf.shared(X)
    H = states(X).reshape(-1, 16); U = pairs(X, sd_tc)[0].reshape(-1, 17)
    roles = {"v19": rstates(T(build_v19(kappa=0.5).params), X)[0].reshape(-1, 16), "v21": rstates(T(build_v21()), X)[0].reshape(-1, 28),
             "v7z": rstates(nets()["v7 + z + ninth slot (18)"], X)[0].reshape(-1, 18)}
    def best(R, Z):
        R = erf.resid(R[:, R.std(0) > 1e-9], F); Z = erf.resid(Z[:, Z.std(0) > 1e-9], F)
        a = (R - R.mean(0)) / (R.std(0) + 1e-12); b = (Z - Z.mean(0)) / (Z.std(0) + 1e-12); bm = (a.T @ b / len(a)).abs().max(1).values
        bm = bm[R.std(0) > 1e-6 * R.std(0).max()]
        return f"best {bm.mean():.2f}, above 0.9 {(bm > 0.9).sum().item()}, above 0.7 {(bm > 0.7).sum().item()} of {len(bm)}"
    dicts = {"ARC's 16 neurons": H}
    for m, k in [(32, 4), (128, 8)]:
        with torch.no_grad(): dicts[f"SAE {m}x k{k}"] = load_sae(m, k).encode(H.float() / sd_sae).to(dt)
    for m in (16, 32, 64, 128):
        for k in (2, 4, 8):
            with torch.no_grad(): dicts[f"TC {m}x k{k}"] = load_tc(m, k).encode(U).to(dt)
    for nm, Z in dicts.items():
        print(f"{nm:18s} | " + " | ".join(f"{rn}: {best(R, Z)}" for rn, R in roles.items()), flush=True)

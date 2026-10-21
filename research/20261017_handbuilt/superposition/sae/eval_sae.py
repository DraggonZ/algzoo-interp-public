"""Evaluate the SAEs. (1) Faithfulness: ARC with its state replaced by the SAE's reconstruction at every step (errors
compound) or at the last step only; accuracy and ARC's mistakes reproduced (grid). (2) Roles: on random lists, pooled
over steps, for each hand-built role (v19 16 neurons, v21 28, v7z 18) the largest |correlation| with any SAE latent
(mean over roles, and how many roles have a latent above 0.9), and the share of each role's variance a linear map from
all latents explains (held out). Baseline dictionary: ARC's own 16 neurons."""
import sys, numpy as np, torch
sys.path.insert(0, "../../family"); sys.path.insert(0, "..")
from common import *
import score
from train_sae import SAE, states
from role_map import states as rstates
from reverse_map import ridge_r2
from baseline_v7z import nets
from algzoo.handbuilt_v19 import build_v19
from algzoo.handbuilt_v21 import build_v21
D = torch.load("saes.pt"); sd = D["sd"].to(DEV)
def load(m, k):
    s = SAE(m, k); s.load_state_dict({kk: v.to(DEV) for kk, v in D[(m, k)].items()}); return s
def spliced(s, every=True):
    def f(X):
        h = torch.zeros(len(X), 16, device=DEV, dtype=dt)
        for t in range(10):
            h = torch.relu(h @ ARC[1].T + X[:, t:t + 1] * ARC[0])
            if every or t == 9:
                with torch.no_grad(): h = (s((h.float() / sd))[0] * sd).to(dt)
        return h @ ARC[2].T
    return f
if __name__ == "__main__":
    pick = [(16, 4), (32, 4), (64, 4), (128, 4), (32, 8), (128, 8), (128, 2)]
    models = {}
    for m, k in pick:
        s = load(m, k); models[f"SAE {m}x k{k}, every step"] = [spliced(s, True)]; models[f"SAE {m}x k{k}, last step"] = [spliced(s, False)]
    score.report(models, family=False)
    g = torch.Generator(device=DEV).manual_seed(77); X = torch.randn(100000, 10, device=DEV, dtype=dt, generator=g); tr, te = slice(0, 50000), slice(50000, None)
    H = states(X); Hf = H.reshape(-1, 16)
    roles = {"v19": rstates(T(build_v19(kappa=0.5).params), X)[0], "v21": rstates(T(build_v21()), X)[0], "v7z": rstates(nets()["v7 + z + ninth slot (18)"], X)[0]}
    def compare(nm, Z):
        Zf = Z.reshape(len(X), 10, -1)
        live = Zf.reshape(-1, Zf.shape[2]).std(0) > 1e-9; Zf = Zf[..., live]
        out = []
        for rn, R in roles.items():
            Rf = R.reshape(-1, R.shape[2]); Rl = Rf[:, Rf.std(0) > 1e-9]; Zc = Zf.reshape(-1, Zf.shape[2])
            a = (Rl - Rl.mean(0)) / Rl.std(0); b = (Zc - Zc.mean(0)) / Zc.std(0); C = (a.T @ b / len(a)).abs()
            best = C.max(1).values
            r2 = ridge_r2(Zf[tr].reshape(-1, Zf.shape[2]), Zf[te].reshape(-1, Zf.shape[2]), R[tr].reshape(-1, R.shape[2])[:, Rf.std(0) > 1e-9], R[te].reshape(-1, R.shape[2])[:, Rf.std(0) > 1e-9])
            out.append(f"{rn}: best match {best.mean():.2f}, roles above 0.9 {(best > 0.9).sum().item()}/{len(best)}, linear R2 {r2.mean():.2f}")
        print(f"{nm:22s} ({live.sum().item():3d} live) | " + " | ".join(out), flush=True)
    compare("ARC's 16 neurons", H)
    for m, k in pick:
        with torch.no_grad(): Z = load(m, k).encode((Hf.float() / sd)).to(dt)
        compare(f"SAE {m}x k{k}", Z)

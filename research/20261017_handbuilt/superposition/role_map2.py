"""Second test: fit each trained neuron's activity (after its ReLU) at every step as one linear map, shared over steps,
from a candidate's activations plus the current number (no intercept: the trained network has no biases, and the
candidates' activity is zero before any number arrives). Held-out R^2 per neuron, roles needed for 90%, and the output
test: ARC's own readout applied to the mapped final activity, scored by accuracy and ARC's mistakes reproduced on the
grid (../family/score.py). Candidates as in role_map.py."""
import sys, numpy as np, torch
sys.path.insert(0, "../family")
from common import *
import score
from role_map import states, plain, random_net
from algzoo.handbuilt_v18 import build_v18
from algzoo.handbuilt_v21 import build_v21
def feats(kind, X):
    if kind == "plain": F = plain(X)
    else: F = states(kind, X)[0]
    return torch.cat([F, X[:, :, None]], 2)
if __name__ == "__main__":
    g = torch.Generator(device=DEV).manual_seed(77); X = torch.randn(200000, 10, device=DEV, dtype=dt, generator=g)
    Ht, _ = states(ARC, X); tr, te = slice(0, 100000), slice(100000, None)
    C = {"v21 (28 roles)": T(build_v21()), "v18 (16)": T(build_v18().params), "random 28, seed 0": random_net(0), "random 28, seed 1": random_net(1), "plain features (12)": "plain"}
    models = {}
    for nm, kind in C.items():
        F = feats(kind, X); sd = F[tr].reshape(-1, F.shape[2]).std(0) + 1e-12; F = F / sd
        Atr = F[tr].reshape(-1, F.shape[2]); Ate = F[te].reshape(-1, F.shape[2])
        G = Atr.T @ Atr; G += 1e-8 * torch.trace(G) / len(G) * torch.eye(len(G), device=DEV, dtype=dt)
        Wm = torch.linalg.solve(G, Atr.T @ Ht[tr].reshape(-1, 16))           # [features, 16]
        P = Ate @ Wm; Y = Ht[te].reshape(-1, 16)
        r2 = 1 - ((P - Y) ** 2).mean(0) / Y.var(0)
        # roles needed: greedy on a subsample, per neuron
        sub = torch.arange(0, len(Atr), 13, device=DEV); As = Atr[sub]; nro = []
        for j in range(16):
            ys = Ht[tr].reshape(-1, 16)[sub, j]; chosen = []; target = 0.9 * max(r2[j].item(), 1e-9)
            while True:
                best, bc = -9, None
                for c in range(As.shape[1]):
                    if c in chosen: continue
                    M = As[:, chosen + [c]]; w = torch.linalg.lstsq(M, ys[:, None]).solution
                    rr = (1 - ((M @ w)[:, 0] - ys).pow(2).mean() / ys.var()).item()
                    if rr > best: best, bc = rr, c
                chosen.append(bc)
                if best >= target or len(chosen) == As.shape[1]: break
            nro.append(len(chosen))
        print(f"{nm:22s} | R2 activity by trained neuron: " + " ".join(f"{v:.2f}" for v in r2.tolist()) + f" | mean {r2.mean():.3f} | roles for 90%: " + " ".join(map(str, nro)), flush=True)
        def mapped(Xg, kind=kind, Wm=Wm, sd=sd):
            out = []
            for i in range(0, len(Xg), 200000):
                Fg = feats(kind, Xg[i:i + 200000])[:, -1] / sd; out.append((Fg @ Wm) @ ARC[2].T)
            return torch.cat(out)
        models[nm + " -> ARC readout"] = [mapped]
    models["ARC itself"] = [ARC]
    score.report(models)

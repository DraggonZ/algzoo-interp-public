"""Controls for the first-order noise prediction (noise_joint.py): networks we built and know completely.
For each network and sigma: true mean accuracy over 200 noisy copies (every weight times 1 + sigma z), each scored on
the same 20,000 lists; prediction from the joint first-order calculation on those 20,000 lists. Networks: the trained
network (reference), the robust twin v21 (28 neurons), v18 at the trained gain scale (g = 10), v18 with clock gain 50."""
import numpy as np
import torch as th
from torch.func import vmap, grad
from algzoo.core import example
from algzoo.handbuilt_v18 import build_v18
from algzoo.handbuilt_v21 import build_v21
dev = "cuda"; th.manual_seed(0)
g = th.Generator(device=dev).manual_seed(11); N = 20000
X = th.randn(N, 10, device=dev, generator=g, dtype=th.float64); y = X.argsort(-1)[:, -2]
nets = {"trained": example(), "robust twin v21": build_v21(),
        "v18 gains 10": build_v18(g=10, a=10, w=10, lam=10.0, c=-9.0, eps=0.1).params,
        "v18 clock gain 50": build_v18(g=50, a=10, w=10, lam=10.0, c=-9.0, eps=0.1).params}
def logits(params, x):
    wi, wh, wo = params; h = th.zeros(len(wi), dtype=x.dtype, device=x.device)
    for t in range(10): h = th.relu(wh @ h + wi * x[t])
    return wo @ h
def batch_acc(P):
    wi, wh, wo = P; h = th.zeros(N, len(wi), dtype=th.float64, device=dev)
    for t in range(10): h = th.relu(h @ wh.T + X[:, t:t + 1] * wi)
    return ((h @ wo.T).argmax(1) == y).double().mean().item()
jac = vmap(grad(lambda p, x, e: (logits(p, x) * e).sum()), in_dims=(None, 0, 0))
oy = th.nn.functional.one_hot(y, 10).double(); keep = th.arange(10, device=dev).unsqueeze(0) != y.unsqueeze(1)
for name, P in nets.items():
    W = [th.as_tensor(np.asarray(w, float), dtype=th.float64, device=dev) for w in P]
    with th.no_grad(): L = vmap(logits, in_dims=(None, 0))(W, X)
    D = ((L * oy).sum(1, keepdim=True) - L)[keep].reshape(-1, 9); C = th.zeros(N, 10, 10, device=dev, dtype=th.float64)
    for b in range(0, N, 5000):
        sl = slice(b, b + 5000)
        G = th.stack([th.cat([(gi * wi.unsqueeze(0)).reshape(5000, -1) for gi, wi in zip(jac(W, X[sl], oy[sl] - th.nn.functional.one_hot(th.full((5000,), k, device=dev), 10).double()), W)], 1) for k in range(10)], 1)
        C[sl] = G @ G.transpose(1, 2)
    C9 = C[keep.unsqueeze(2) & keep.unsqueeze(1)].reshape(-1, 9, 9)
    out = [f"no noise {batch_acc(W):.3f}"]
    for sig in (0.005, 0.01, 0.02):
        accs = [batch_acc([w * (1 + sig * th.randn(w.shape, device=dev, dtype=th.float64, generator=g)) for w in W]) for _ in range(200)]
        Lc = th.linalg.cholesky(C9 * sig ** 2 + 1e-9 * th.eye(9, device=dev, dtype=th.float64))
        pr = []
        for b in range(0, N, 5000):
            Z = th.randn(5000, 2000, 9, device=dev, dtype=th.float64, generator=g)
            pr.append(((D[b:b + 5000].unsqueeze(1) + Z @ Lc[b:b + 5000].transpose(1, 2)) > 0).all(2).double().mean(1))
        out.append(f"sigma {sig}: true {np.mean(accs):.3f} (copies SD {np.std(accs):.3f}), predicted {th.cat(pr).mean().item():.3f}")
    print(f"{name:18s} " + " | ".join(out), flush=True)

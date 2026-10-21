"""Three-input reduction (review 5): keep the top k inputs, set the rest to a background value; how close is the
reduced accuracy to the true accuracy? Sampled diagnostic (2^23 fresh inputs per model).

Backgrounds: conditional mean E[X | X < k-th largest] = -phi(m_k)/Phi(m_k) per input, and the constant -0.46 (mean of the
seven smallest of ten normals). Target: the original second's position.
"""
import numpy as np
import torch as th

from algzoo.bench import load_bench
from algzoo.core import example, zoo

dev = "cuda"
N = 2 ** 23


def accs(params, X, k):
    w = [th.as_tensor(q, device=dev, dtype=th.float32) for q in params]
    o = X.argsort(-1)
    tgt = o[:, -2]
    mk = X.gather(1, o[:, -k:-k + 1] if k > 1 else o[:, -1:])[:, 0]
    nd = th.distributions.Normal(0.0, 1.0)
    cm = -th.exp(nd.log_prob(mk)) / nd.cdf(mk).clamp_min(1e-12)
    out = {}
    topk = o[:, -k:]
    for lab, bgv in [("full", None), ("cond-mean", cm), ("const -0.46", th.full_like(mk, -0.46))]:
        if bgv is None:
            Xr = X
        else:
            Xr = bgv[:, None].expand_as(X).clone()
            Xr.scatter_(1, topk, X.gather(1, topk))
        h = th.zeros(N, 16, device=dev)
        for t in range(10):
            h = th.relu(h @ w[1].T + Xr[:, t, None] * w[0])
        out[lab] = ((h @ w[2].T).argmax(-1) == tgt).float().mean().item()
    return out


g = th.Generator(device=dev).manual_seed(77)
X = th.randn(N, 10, device=dev, generator=g)
names, acc, models = load_bench(("mult",))
sig = np.array([float(str(n).split("_")[0][4:]) for n in names])
for k in (3, 4):
    rows = []
    for name, p in [("example", example())] + [(f"seed{s}", zoo(16, 10, s)) for s in range(5)]:
        o = accs(p, X, k)
        rows.append((o["full"], o["cond-mean"], o["const -0.46"]))
        print(f"k={k} {name}: full {o['full']:.4f}, cond-mean {o['cond-mean']:.4f} ({100*(o['cond-mean']-o['full']):+.1f}), const {o['const -0.46']:.4f} ({100*(o['const -0.46']-o['full']):+.1f})", flush=True)
    R = np.array(rows)
    print(f"k={k} trained RMSE (points): cond-mean {100*np.sqrt(np.mean((R[:,1]-R[:,0])**2)):.2f}, const {100*np.sqrt(np.mean((R[:,2]-R[:,0])**2)):.2f}")
    Xs = X[: 2 ** 21]
    rows = []
    for nm, p in zip(names, models):
        global_N = N
        o = accs(p, X, k)
        rows.append((o["full"], o["cond-mean"], o["const -0.46"]))
    R = np.array(rows)
    for s in (0.005, 0.01, 0.02):
        kk = sig == s
        print(f"k={k} copies sigma {s}: RMSE cond-mean {100*np.sqrt(np.mean((R[kk,1]-R[kk,0])**2)):.2f}, const {100*np.sqrt(np.mean((R[kk,2]-R[kk,0])**2)):.2f} (bias {100*np.mean(R[kk,2]-R[kk,0]):+.2f})", flush=True)
    print(f"k={k} copies all: RMSE cond-mean {100*np.sqrt(np.mean((R[:,1]-R[:,0])**2)):.2f}, const {100*np.sqrt(np.mean((R[:,2]-R[:,0])**2)):.2f}")

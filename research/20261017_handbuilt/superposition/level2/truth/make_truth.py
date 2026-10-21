"""Shared truth for Level 2: noisy copies of ARC's network (every one of the 432 weights + N(0, s^2)), s = 0.001, 0.0025,
0.005, 0.01, 100 copies each, the noise vectors saved (order: w_hi (16), W_hh (256, row-major), W_oh (160, row-major)).
True accuracy on 2,000,000 fixed random lists (standard error about 0.015 points), and the shares of the copy's
mistakes that name the largest, the third, or another position. Also ARC itself on the same lists."""
import sys, numpy as np, torch
sys.path.insert(0, "../../../family")
from common import *
if __name__ == "__main__":
    wi, wh, wo = ARC; w0 = torch.cat([wi.flatten(), wh.flatten(), wo.flatten()])
    def unflat(v): return [v[:16], v[16:272].reshape(16, 16), v[272:].reshape(10, 16)]
    g = torch.Generator(device=DEV).manual_seed(2026); X = torch.randn(2000000, 10, device=DEV, dtype=dt, generator=g)
    o = X.argsort(1, descending=True); L, S, Tq = o[:, 0], o[:, 1], o[:, 2]
    def stats(W):
        a = torch.cat([scores(W, X[i:i + 500000]).argmax(1) for i in range(0, len(X), 500000)])
        wr = a != S; n = wr.sum().item()
        return [(~wr).double().mean().item(), (a[wr] == L[wr]).double().sum().item() / max(n, 1), (a[wr] == Tq[wr]).double().sum().item() / max(n, 1)]
    out = {"arc": np.array(stats(ARC))}
    print(f"ARC: accuracy {100*out['arc'][0]:.3f}%, mistakes naming the largest {100*out['arc'][1]:.1f}%, the third {100*out['arc'][2]:.1f}%")
    gn = torch.Generator(device=DEV).manual_seed(77)
    for s in (0.001, 0.0025, 0.005, 0.01):
        dws = torch.randn(100, 432, device=DEV, dtype=dt, generator=gn) * s; res = np.array([stats(unflat(w0 + d)) for d in dws])
        out[f"dw_{s}"] = dws.cpu().numpy(); out[f"stats_{s}"] = res
        print(f"s {s}: accuracy mean {100*res[:,0].mean():.2f}%, SD {100*res[:,0].std():.2f}, range {100*res[:,0].min():.1f}-{100*res[:,0].max():.1f}; mistakes naming the largest {100*res[:,1].mean():.1f}%, the third {100*res[:,2].mean():.1f}%", flush=True)
    np.savez("noisy_truth.npz", **out)

"""How large is ARC's leak in decisions, and what shape? On random lists, the exact lead map d = J[s] - J[t] (scores are
linear on each list) scaled by the second's own coefficient. For every other number o: |d_o| by its role (the largest,
a positive smaller number, a negative number) and by its offset from the second. Same for v23 and v23 with the forced
nine-in-eight leak. Question: which numbers carry ARC's leak, and how big is it per unit?"""
import sys; sys.path.insert(0, "/data/output/research/algzoo/20261017_handbuilt/family"); sys.path.insert(0, "../v23")
import numpy as np, torch
from common import DEV, dt, jac, top3, T, ARC
from algzoo.handbuilt_v23 import build_v23
from leak_net import leaky
from nine_in_eight import null_dir
g = torch.Generator(device=DEV).manual_seed(77); X = torch.randn(100_000, 10, device=DEV, dtype=dt, generator=g); L, S, T3 = top3(X); n = torch.arange(len(X), device=DEV)
def shape(W, name):
    J = jac(W, X); d = J[n, S] - J[n, T3]; d = d / d[n, S][:, None]
    pos = torch.arange(10, device=DEV)[None].expand(len(X), 10)
    other = (pos != S[:, None]) & (pos != T3[:, None]); isL = pos == L[:, None]
    roles = {"the largest": other & isL, "positive smaller": other & ~isL & (X > 0), "negative": other & (X <= 0)}
    print(f"{name}: coefficient of the third on the lead, median {d[n, T3].median():+.2f}")
    for k, m in roles.items():
        v = d[m].abs(); print(f"   {k:17s} |d| median {v.median():.3f}, 90% {v.quantile(0.9):.3f}, share above 0.02 {(v > 0.02).double().mean():.2f}")
    off = (pos - S[:, None]); m = other & ~isL & (X > 0)
    print("   positive smaller, median |d| by offset from the second (-4..4): " + " ".join(f"{d[m & (off == k)].abs().median():.3f}" for k in range(-4, 5) if k != 0))
if __name__ == "__main__":
    shape(ARC, "ARC")
    shape(T(build_v23().params), "v23")
    shape(T(leaky(build_v23(), null_dir([0, 43, 83, 126, 180], np.array([0.99, 0.94, 0.94, 0.82, 0.90]))).params), "v23 + ARC leak")

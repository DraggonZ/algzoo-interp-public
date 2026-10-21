"""w12b analysis 0: in w8b, what the linear route reads and who reads it; the keep gate's row; accuracy base."""
import numpy as np
from algzoo.handbuilt_w8b import build_w8b

np.set_printoptions(precision=4, suppress=False, linewidth=160)
wi, wh, wo, idx, _ = build_w8b()
name = {v: k for k, v in idx.items()}
print("neurons", len(wi), idx)
for nm in ("linear", "keep", "store", "mem0", "mem1"):
    if nm not in idx: continue
    n = idx[nm]
    print(f"\n{nm}: wi={wi[n]:.4g}  reads:", {name[k]: float(f"{wh[n, k]:.4g}") for k in np.flatnonzero(np.abs(wh[n]) > 1e-12)})
    print(f"  read by:", {name[k]: float(f"{wh[k, n]:.4g}") for k in np.flatnonzero(np.abs(wh[:, n]) > 1e-12)})
    print("  readout:", np.round(wo[:, n], 4))


def acc(wi, wh, wo, n=200000, seed=1):
    x = np.random.default_rng(seed).standard_normal((n, 10)); h = np.zeros((n, len(wi)))
    for t in range(10):
        h = np.maximum(h @ wh.T + x[:, t:t + 1] * wi, 0)
    tgt = np.argsort(x, 1)[:, -2]
    return (np.argmax(h @ wo.T, 1) == tgt).mean()

print("\nbase accuracy", acc(wi, wh, wo))

"""Review 2 check (sampling-based diagnostic, not an estimator).

Question: can a zero-background statistic with ONE constant taken from the base model track the copies?
(1) Z_2(tau): fraction of inputs whose top-two-only logit margin (target minus best other) exceeds tau,
    with tau set so that the base model's value equals its true accuracy 0.9531.
(2) The same with top-three-only logits (Z_3(tau)).
(3) Additive surrogate with a ratio from the base: true acc ~ additive acc * (0.9531 / base additive acc).
Fits nothing to the copies. 2^18 inputs per model.
"""
import numpy as np
import torch as th

from algzoo.bench import load_bench
from algzoo.core import example, forward

dev = "cuda"
N = 2**18
BASE_ACC = 0.9531


def margin(L, p2):
    lt = L.gather(1, p2[:, None])[:, 0]
    return lt - L.scatter(1, p2[:, None], -th.inf).max(-1).values


@th.no_grad()
def stats(params, seed):
    g = th.Generator(device=dev).manual_seed(seed)
    x = th.randn(N, 10, generator=g, device=dev, dtype=th.float64)
    v, idx = x.sort(-1, descending=True)
    p2 = idx[:, 1]
    out = {}
    for k in (2, 3):
        xk = th.zeros_like(x).scatter_(1, idx[:, :k], v[:, :k])
        out[f"m{k}"] = margin(forward(params, xk), p2).cpu().numpy()
    top2 = th.zeros_like(x).scatter_(1, idx[:, :2], v[:, :2])
    L0 = forward(params, top2)
    Ladd = L0.clone()
    for k in range(8):
        xs = top2.clone().scatter_(1, idx[:, 2 + k:3 + k], v[:, 2 + k:3 + k])
        Ladd += forward(params, xs) - L0
    out["add"] = (Ladd.argmax(-1) == p2).double().mean().item()
    out["full"] = (forward(params, x).argmax(-1) == p2).double().mean().item()
    return out


base = stats(example(), 0)
taus = {}
for k in (2, 3):
    m = base[f"m{k}"]
    taus[k] = np.quantile(m, 1 - BASE_ACC)
    print(f"base: Z_{k} = {(m > 0).mean():.4f}; tau for {BASE_ACC}: {taus[k]:.4f}")
print(f"base: additive {base['add']:.4f}, full (sampled) {base['full']:.4f}; ratio {BASE_ACC / base['add']:.4f}")

names, acc, models = load_bench(("mult",))
sig = np.array([float(str(n).split("_")[0][4:]) for n in names])
rows = []
for i, m in enumerate(models):
    s = stats(m, 1 + i)
    rows.append([acc[i], (s["m2"] > 0).mean(), (s["m2"] > taus[2]).mean(), (s["m3"] > 0).mean(), (s["m3"] > taus[3]).mean(),
                 s["add"], s["add"] * BASE_ACC / base["add"]])
rows = np.array(rows)
np.save("margin_threshold.npy", rows)
t = rows[:, 0]
for j, nm in enumerate(["Z2", "Z2(tau from base)", "Z3", "Z3(tau from base)", "additive", "additive x base ratio"]):
    e = rows[:, 1 + j]
    parts = [f"{nm:22s} RMSE {np.sqrt(np.mean((e - t) ** 2)):.4f} bias {np.mean(e - t):+.4f} corr {np.corrcoef(e, t)[0, 1]:.3f}"]
    for s in (0.005, 0.01, 0.02):
        k = sig == s
        parts.append(f"σ={s}: {np.sqrt(np.mean((e[k] - t[k]) ** 2)):.4f} ({np.mean(e[k] - t[k]):+.3f})")
    print(" | ".join(parts))

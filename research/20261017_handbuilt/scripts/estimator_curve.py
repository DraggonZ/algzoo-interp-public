"""Error of the bias-size formula against compute, compared with random sampling, on 38 models
(22 trained + 16 fresh seeds 200-215). Bias per placement from one run + derivative (clipped).
Knob: K placements, a fixed evenly spread subset of the 720 (no randomness); G values of the largest.
Cost counted as network runs: (forward + backward ~ 3 runs) x K x (G + 1)."""
import itertools
import json
from pathlib import Path

import numpy as np
import torch as th

from algzoo.core import example, forward, load, zoo

PL = np.array(list(itertools.permutations(range(10), 3)))
C23, C12, ONE = 1.926, 1.719, 11 / 1024 * 8 / 9


def b_one_run(P, pl, which, m1):
    N = len(pl); r = np.arange(N)
    x = np.full((N, 10), -0.46); x[r, pl[:, 0]] = m1
    if which == "23":
        x[r, pl[:, 1]] = 1.0; x[r, pl[:, 2]] = 1.0 - 1e-6; a, b, var, scale = 1, 2, 2, 1.0
    else:
        x[r, pl[:, 1]] = m1 * (1 - 1e-6); x[r, pl[:, 2]] = 0.5; a, b, var, scale = 1, 0, 1, m1
    X = th.tensor(x, device="cuda", requires_grad=True)
    s = forward(P, X); M = s[r, pl[:, a]] - s[r, pl[:, b]]
    g = th.autograd.grad(M.sum(), X)[0][r, pl[:, var]]
    step = np.nan_to_num((-(M / g)).detach().cpu().numpy(), nan=0.0, posinf=1, neginf=-1)
    return np.clip(step / scale, -0.2, 0.2)


rng = np.random.default_rng(1)
x = np.sort(rng.standard_normal((5_000_000, 10)), 1)
ratio = (x[:, -1] / x[:, -2])[(x[:, -2] > 0) & (x[:, -2] - x[:, -3] < 0.01 * x[:, -2])]


def grid_weights(G):
    qs = np.percentile(ratio, (np.arange(G) + 0.5) / G * 100)  # G equal-mass bins, value at each bin's median
    return qs, np.full(G, 1 / G)


def estimate(P, K, G):
    idx = np.linspace(0, len(PL) - 1, K).round().astype(int)
    pl = PL[idx]
    qs, w = grid_weights(G)
    a = sum(wi * np.mean(np.maximum(0, -b_one_run(P, pl, "23", q))) for q, wi in zip(qs, w))
    b = np.mean(np.maximum(0, -b_one_run(P, pl, "12", 1.6)))
    return ONE + C23 * a + C12 * b


def main():
  pass
if __name__ != "__main__":
    raise SystemExit
models = {"ARC": example(), **{f"zoo{s}": zoo(16, 10, s) for s in range(5)}}
for f in sorted(Path("/data/output/research/algzoo/20261016_run3/models").glob("2nd_argmax_16_10_*.pth")):
    models["s" + f.stem.split("_")[-1]] = load(str(f))
truth = {}
for src in ("bias_width_rows.json", "bias_formula_v2_fresh.json"):
    for k, v in json.load(open(src)).items():
        if "acc" in v:
            truth[{"new": "s"}.get(k[:3], "") + k[3:] if k.startswith("new") else k] = 1 - v["acc"]
        elif "err" in v:
            truth["s" + k[5:]] = v["err"]
names = [k for k in models if k in truth]
err = np.array([truth[k] for k in names])
print(f"{len(names)} models, error rates {100*err.min():.1f}-{100*err.max():.1f} pts; constant guess misses by {100*err.std():.2f} pts")
for K, G in ((720, 7), (720, 3), (240, 3), (90, 3), (30, 3), (30, 1), (10, 1)):
    pred = np.array([estimate(models[k], K, G) for k in names])
    rmse = 100 * np.sqrt(np.mean((pred - err) ** 2))
    runs = 3 * K * (G + 1)
    samp = 100 * np.sqrt(np.mean(err * (1 - err) / runs))
    print(f"K={K:3d} G={G}: cost {runs:6d} runs; miss {rmse:.2f} pts (without seed 208: {100*np.sqrt(np.mean(np.delete(pred - err, names.index('s208'))**2)):.2f}); sampling at the same cost {samp:.2f} pts; squared-error ratio {(rmse/samp)**2:.1f}")

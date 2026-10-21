"""Are second-third tie biases shared across trained models? For each of the 720 placements of the
top three (largest 1.6, second 1.0, others -0.46), find the third's value at which the model's
choice between the second's and third's slots flips (bisection in [0.8, 1.2]). Bias = flip value - 1.
Correlate biases across ARC's model, zoo seeds 0-4 and run 3's 16 new models."""
import itertools
from pathlib import Path

import numpy as np
import torch as th

from algzoo.core import example, forward, load, zoo

models = {"ARC": example()}
for s in range(5):
    models[f"zoo{s}"] = zoo(16, 10, s)
for f in sorted(Path("/data/output/research/algzoo/20261016_run3/models").glob("2nd_argmax_16_10_*_1*.pth")):
    models["new" + f.stem.split("_")[-1]] = load(str(f))
pl = np.array(list(itertools.permutations(range(10), 3)))
N = len(pl); r = np.arange(N)
base = np.full((N, 10), -0.46); base[r, pl[:, 0]] = 1.6; base[r, pl[:, 1]] = 1.0
B = {}
for name, P in models.items():
    lo, hi = np.full(N, 0.8), np.full(N, 1.2)
    for _ in range(30):
        mid = (lo + hi) / 2
        x = base.copy(); x[r, pl[:, 2]] = mid
        s = forward(P, th.tensor(x, device="cuda")).cpu().numpy()
        third_wins = s[r, pl[:, 2]] > s[r, pl[:, 1]]
        hi = np.where(third_wins, mid, hi); lo = np.where(third_wins, lo, mid)
    B[name] = (lo + hi) / 2 - 1
names = list(B)
M = np.array([B[k] for k in names])
print(f"{len(names)} models; typical |bias| per model (median): " + ", ".join(f"{k} {np.median(np.abs(B[k])):.3f}" for k in names[:6]) + " ...")
C = np.corrcoef(M)
off = C[~np.eye(len(names), dtype=bool)]
print(f"correlation of bias patterns between models: median {np.median(off):.2f}, range {off.min():.2f} to {off.max():.2f}")
print("ARC vs others:", " ".join(f"{k}:{C[0, i]:.2f}" for i, k in enumerate(names) if i))
# systematic part: average over other models; does it predict each model?
for i, k in enumerate(names[:6]):
    others = np.delete(M, i, 0).mean(0)
    print(f"  {k}: correlation with the average of the other models {np.corrcoef(M[i], others)[0,1]:.2f}")
# which simple factors carry the shared part? order of arrival
mean_all = M.mean(0)
order = np.array([("2<3" if a < b else "3<2") + (" L first" if c < min(a, b) else (" L last" if c > max(a, b) else " L mid")) for c, a, b in pl])
for o in sorted(set(order)):
    print(f"  {o:12s}: mean bias across models {mean_all[order==o].mean():+.4f}  (positive = favours the second)")
np.save("tie_bias_models.npy", M)

"""Question: do the other trained networks also keep an overtaken record's value off the positions after the overtaking?
Role-free check (no neuron labels needed). For each network (as bridge/second_comparator_across.py: ARC, zoo0-4, and the
run-3 networks new100-115), lists with a record at r overtaken at q (as tail_route.py), all placements with q <= 8.
Nudge x_r and x_q together (jump fixed) and x_r alone; per unit, mean over lists, then over placements:
  at r (the revealed vote), the largest per-unit effect at any one position q+1..8 (worst placement and mean over
  placements of each placement's largest), the mean effect at positions q+1..8, and position 9.
400,000 lists per network, seed 14."""
from pathlib import Path
import numpy as np
from algzoo.core import example, zoo, load

models = {"ARC": example()}
for s in range(5): models[f"zoo{s}"] = zoo(16, 10, s)
for f in sorted(Path("/data/output/research/algzoo/20261016_run3/models").glob("2nd_argmax_16_10_*_1*.pth")):
    models["new" + f.stem.split("_")[-1]] = load(str(f))
X = np.random.default_rng(14).standard_normal((400_000, 10)); eps = 1e-4
pm = np.maximum.accumulate(np.c_[np.zeros(len(X)), X], 1)


def scores(P, X):
    wi, wh, wo = P; h = np.zeros((len(X), len(wi)))
    for t in range(10): h = np.maximum(h @ wh.T + X[:, t:t + 1] * wi, 0)
    return h @ wo.T


if __name__ == "__main__":
  print("network  acc  | nudge both: at r, worst single later position, mean of each placement's largest later, mean later, pos 9 | nudge r: same")
  for name, w in models.items():
      P = [np.asarray(p, float) for p in w]; P[0] = P[0].reshape(-1)
      acc = (scores(P, X).argmax(1) == X.argsort(1)[:, -2]).mean()
      out = {}
      for nudge in ("both", "r"):
          at_r, worst, mx, later, p9 = [], [], [], [], []
          for q in range(2, 9):
              for r in range(0, q):
                  ok = (X[:, r] > pm[:, r] + 0.05) & (X[:, r] > 0.3) & (X[:, q] > X[:, r] + 0.05) & (X[:, q + 1:].max(1) < X[:, q])
                  if q > r + 1: ok &= X[:, r + 1:q].max(1) < X[:, r]
                  Xs = X[ok]
                  if len(Xs) < 500: continue
                  dX = np.zeros_like(Xs); dX[:, r] = eps
                  if nudge == "both": dX[:, q] = eps
                  d = ((scores(P, Xs + dX) - scores(P, Xs)) / eps).mean(0)
                  at_r.append(d[r]); p9.append(d[9])
                  if q < 8: mx.append(d[q + 1:9].max()); later.append(d[q + 1:9].mean())
          out[nudge] = f"{np.mean(at_r):6.1f} {np.max(mx):6.1f} {np.mean(mx):6.1f} {np.mean(later):6.1f} {np.mean(p9):6.1f}"
      print(f"{name:7s} {100 * acc:5.1f} | {out['both']} | {out['r']}")

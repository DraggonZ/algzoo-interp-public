"""Follow-up to clip_recovery.py: at a partial switch (steps 3-6, largest at step 2), which hiding neurons clip, and how
large is each one's sensitivity to the largest in the counterfactual list (the switching number replaced by 0), relative to
the group's? Also the clipped neurons' share of the group's readout of the largest's score (d score[r] / d x_r through
that neuron, counterfactual). Trained network and v18 at g = 10."""
import numpy as np
from algzoo.core import example
from algzoo.handbuilt_v18 import build_v18
rng = np.random.default_rng(12); X = rng.standard_normal((1500000, 10))
def states(P, X):
    P = [np.asarray(p, float) for p in P]; h = np.zeros((len(X), len(P[0]))); hs = []
    for t in range(10): h = np.maximum(h @ P[1].T + X[:, t:t + 1] * P[0], 0); hs.append(h)
    return np.stack(hs, 1)
I = build_v18().idx; hid18 = [I[k] for k in ("clock0", "clock1", "clock2", "flip")]
nets = {"trained": (example(), [1, 6, 7, 8]), "v18 g 10": (build_v18(g=10, a=10, w=10, lam=10.0, c=-9.0, eps=0.1).params, hid18)}
import sys
r = 2; eps = 1e-3
for c in (3, 4, 5, 6):
  base = (X[:, r] > np.maximum(X[:, :r].max(1), 0) + 0.05) & (X[:, r] > 0.3) & (X[:, r + 1:].max(1) < X[:, r]) & (X[:, c] > 0)
  print(f"--- largest at {r}, partial switch at step {c}")
  for name, (P, hid) in nets.items():
      H = states(P, X[base]); off = (H[:, c, hid] == 0); part = (off.sum(1) > 0) & (off.sum(1) < len(hid))
      Xs = X[base][part][:40000]; offs = off[part][:40000]; Xc = Xs.copy(); Xc[:, c] = 0.0
      Xp = Xc.copy(); Xp[:, r] += eps; Sc = (states(P, Xp) - states(P, Xc))[:, c, :] / eps     # counterfactual, after step 3
      vals = states(P, Xc)[:, c, :]
      s = np.abs(Sc[:, hid]); share = (s * offs).sum(1) / s.sum(1)
      v = vals[:, hid]; vshare = (v * offs).sum(1) / v.sum(1)
      print(f"{name:9s} which neurons clip (share of lists): " + ", ".join(f"{n} {offs[:, i].mean():.2f}" for i, n in enumerate(hid)))
      print(f"          clipped neurons' share of the group's sensitivity to the largest: median {np.median(share):.2f}; of the group's value: {np.median(vshare):.2f}")
      print(f"          sensitivities d h / d x_r (counterfactual), median by neuron: " + ", ".join(f"{n} {np.median(Sc[:, j]):.2f}" for n, j in zip(hid, hid)))

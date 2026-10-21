"""Per-triple diagnosis: sampling-free I(triple) versus sampled per-triple reduced accuracy, for one model."""
import itertools
import sys

import numpy as np
import torch as th

sys.path.insert(0, "/data/output/research/algzoo/20261016_run3/scripts")
from reduce3_det import B, estimate  # noqa: E402

from algzoo.core import example, zoo  # noqa: E402

name = sys.argv[1]
p = example() if name == "example" else zoo(16, 10, int(name))
e, ops, det = estimate(p)
T = np.array(list(itertools.permutations(range(10), 3)))
# recompute per-triple I
import reduce3_det as R  # noqa: E402
from scipy.special import ndtr  # noqa: E402
sc = np.maximum((R.M2[None, :] - B) / (R.L_MID - B), 0)
Mm = det["Mlo"][:, None] + (det["Mhi"] - det["Mlo"])[:, None] * (R.M2[None, :] - R.L_LO) / (R.L_HI - R.L_LO)
C = (Mm > 0).astype(float)
f = 720 * R.PHI2[None, :] * C * (1 - ndtr(R.M2[None, :] + det["w12"][:, None] * sc)) * ndtr(R.M2[None, :] - det["w23"][:, None] * sc) ** 8 / 8
I = (f * R.W2[None, :]).sum(1)
# sampled per triple
dev = "cuda"
w = [th.as_tensor(q, device=dev, dtype=th.float32) for q in p]
g = th.Generator(device=dev).manual_seed(5)
N = 2 ** 22
X = th.randn(N, 10, device=dev, generator=g)
o = X.argsort(-1)
top3 = o[:, -3:]
Xr = th.full_like(X, B); Xr.scatter_(1, top3, X.gather(1, top3))
h = th.zeros(N, 16, device=dev)
for t in range(10):
    h = th.relu(h @ w[1].T + Xr[:, t, None] * w[0])
right = ((h @ w[2].T).argmax(-1) == o[:, -2]).float()
key = (o[:, -1] * 100 + o[:, -2] * 10 + o[:, -3]).cpu().numpy()
rr = right.cpu().numpy()
cnt = np.bincount(key, minlength=1000); s = np.bincount(key, weights=rr, minlength=1000)
samp = np.array([s[a * 100 + b * 10 + c] / max(cnt[a * 100 + b * 10 + c], 1) for a, b, c in T])
diff = I - samp
print(f"{name}: mean I {I.mean():.4f}, mean sampled {samp.mean():.4f}")
idx = np.argsort(diff)[:12]
print("worst triples (largest, second, third): I, sampled, w23, w12, bulk margins at low/high level")
for i in idx:
    print(f"  {tuple(T[i])}: I {I[i]:.3f} sampled {samp[i]:.3f} | w23 {det['w23'][i]:.3f} w12 {det['w12'][i]:.3f} | Mlo {det['Mlo'][i]:+.2f} Mhi {det['Mhi'][i]:+.2f}")
# which component: recompute with C=1 everywhere, and with w12=0, and with w23=0
for lab, CC, a, b_ in [("bulk always right", np.ones_like(C), det["w12"], det["w23"]), ("no 1-2 band", C, 0 * det["w12"], det["w23"]), ("no 2-3 band", C, det["w12"], 0 * det["w23"])]:
    ff = 720 * R.PHI2[None, :] * CC * (1 - ndtr(R.M2[None, :] + a[:, None] * sc)) * ndtr(R.M2[None, :] - b_[:, None] * sc) ** 8 / 8
    print(f"  variant '{lab}': mean {(ff * R.W2[None, :]).sum(1).mean():.4f}")

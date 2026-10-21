"""Which lists does 0.1% noise flip in v20? Clean answer vs noisy answer; for flipped lists: clean lead, what the noisy
answer names, and list structure."""
import numpy as np
from algzoo.handbuilt_v20 import build_v20
import simlib
X = simlib.lists(90, 50_000); y = X.argsort(1)[:, -2]; o = X.argsort(1)[:, ::-1]; rng = np.random.default_rng(0)
P = build_v20(lam=12.75); L, _, H = simlib.run(X, params=P); a0 = L.argmax(1)
typ = np.array([np.median(H[:, :, n][H[:, :, n] > 0]) if (H[:, :, n] > 0).any() else 1.0 for n in range(len(P[0]))])
h = np.zeros((len(X), len(P[0])))
for t in range(10):
    h = np.maximum(h @ P[1].T + X[:, t:t + 1] * P[0], 0); h = np.maximum(h + 1e-3 * typ * rng.standard_normal(h.shape) * (h > 0), 0)
a1 = (h @ P[2].T).argmax(1); fl = a0 != a1
srt = np.sort(L, 1); lead = srt[:, -1] - srt[:, -2]
print(f"flipped {fl.mean():.3f}; clean right among flipped {np.mean(a0[fl] == y[fl]):.2f}; median clean lead flipped {np.median(lead[fl]):.4f} (all {np.median(lead):.3f})")
npos = (X > 0).sum(1)
print(f"flipped lists with at most one positive number {np.mean(npos[fl] <= 1):.2f}; flipped where clean answer is position 8 {np.mean(a0[fl] == 8):.2f}, 9 {np.mean(a0[fl] == 9):.2f}; noisy answer 8 {np.mean(a1[fl] == 8):.2f}, 9 {np.mean(a1[fl] == 9):.2f}")
print("clean lead deciles over all lists:", np.round(np.quantile(lead, [0.05, 0.1, 0.2, 0.3]), 4))
i = np.flatnonzero(fl & (npos > 1))[:3]
np.set_printoptions(precision=3, suppress=True, linewidth=160)
for k in i: print(X[k], "\n  ", L[k] / 10, a0[k], a1[k], y[k])

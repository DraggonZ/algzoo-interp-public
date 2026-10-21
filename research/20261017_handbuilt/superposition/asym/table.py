"""Early versus late asymmetry, controlled. Second at position pS, largest at pL (second ~ |N(0,1)| + 0.5, largest =
second x (1 + |N(0, 0.3)|) + small), the other eight uniform in [0, 0.97 x second) (30% replaced by negatives, as in
compose.py). Accuracy for every (pS, pL). Guesses (written first): H1 the reveal of the second when the largest
arrives late (steps 8-9, after the second) is weak, so accuracy is lowest there; H2 the second's own position matters
(late second, fewer steps to be confirmed)."""
import sys; sys.path.insert(0, "../steer")
import numpy as np
from harness import *
g = torch.Generator(device=DEV).manual_seed(11); N = 100_000
acc = np.full((10, 10), np.nan)
for ps in range(10):
    for pl in range(10):
        if ps == pl: continue
        s = torch.randn(N, device=DEV, dtype=dt, generator=g).abs() + 0.5
        l = s * (1 + 0.3 * torch.randn(N, device=DEV, dtype=dt, generator=g).abs()) + 1e-3
        X = s[:, None] * 0.97 * torch.rand(N, 10, device=DEV, dtype=dt, generator=g)
        X = torch.where(torch.rand(N, 10, device=DEV, dtype=dt, generator=g) < 0.3, -torch.randn(N, 10, device=DEV, dtype=dt, generator=g).abs(), X)
        X[:, ps] = s; X[:, pl] = l
        acc[ps, pl] = 100 * (run(ARC, X).argmax(1) == ps).double().mean().item()
np.set_printoptions(precision=1, suppress=True, linewidth=160)
print("accuracy %, rows: second's position 0-9, columns: largest's position 0-9")
print(acc)
print(f"mean by second's position: {np.round(np.nanmean(acc, 1), 1)}")
print(f"mean by largest's position: {np.round(np.nanmean(acc, 0), 1)}")
up = np.nanmean([acc[i, j] for i in range(10) for j in range(10) if j > i]); lo = np.nanmean([acc[i, j] for i in range(10) for j in range(10) if j < i])
print(f"largest after the second: {up:.1f}%; largest before: {lo:.1f}%")
np.save("table.npy", acc)

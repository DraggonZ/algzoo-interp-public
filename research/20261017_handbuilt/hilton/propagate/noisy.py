"""The mean-background estimate on noisy copies of ARC's network (each weight + N(0, 0.0025^2), ARC's noise),
each copy against its true accuracy on 4M lists. Grid 40 x 48 x 48 (about 0.1 point below the converged value on ARC)."""
import numpy as np, torch
import adf, full_mean, truth
W0 = adf.net(None); g = torch.Generator(device="cuda").manual_seed(5); rows = []
for c in range(40):
    W = tuple(w + 0.0025 * torch.randn(w.shape, device="cuda", dtype=w.dtype, generator=g) for w in W0)
    t = truth.true_acc(W, n=4_000_000, seed=100 + c); e, _ = full_mean.estimate(W, 40, 48); rows.append((t, e))
    print(f"copy {c:2d}: true {100*t:6.2f}%  estimate {100*e:6.2f}%  error {100*(e-t):+.2f}", flush=True)
r = np.array(rows) * 100; err = r[:, 1] - r[:, 0]
print(f"40 copies: true {r[:,0].min():.1f}-{r[:,0].max():.1f}%, RMSE {np.sqrt((err**2).mean()):.2f}, mean error {err.mean():+.2f}, correlation {np.corrcoef(r.T)[0,1]:.3f}")
np.save("noisy.npy", r)

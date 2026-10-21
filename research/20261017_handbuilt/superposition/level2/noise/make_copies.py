"""Noisy copies and their true accuracy (1M shared random lists per network; standard error of one copy about 0.02 points).
ARC: the 200 copies of hilton_noise.py at s = 0.0025 (checked against their stored accuracy on 100,000 lists), plus
fresh copies at s = 0.001 and 0.005. Other trained networks (seeds 106, 111, 115 (four, four and three hiding neurons)): copies at s = 0.0025.
Writes copies.npz: per set the noise draws [C, 432] and true accuracies."""
import numpy as np, torch as th, time
from noiselib import net, arc_hilton_copies, draws, accuracy, flat, unflat, DEV, DT

out = {}
W = net("ARC")
X, dws = arc_hilton_copies()
old = np.load("/data/output/research/algzoo/20261017_handbuilt/hilton/hilton_noise.npz")["true"]
t0 = time.time()
# exact check on the original 100,000 lists
wi, wh, wo = unflat(flat(W)[None] + dws, W)
h = th.zeros(200, len(X), 16, device=DEV, dtype=DT)
for t in range(10): h = th.relu(th.einsum("cbj,cij->cbi", h, wh) + X[None, :, t:t + 1] * wi[:, None, :])
acc100k = (th.einsum("cbj,cij->cbi", h, wo).argmax(-1) == X.argsort(-1)[:, -2][None]).double().mean(1).cpu().numpy()
print(f"regenerated copies: max difference from hilton_noise.npz accuracy {np.abs(acc100k - old).max():.2e}")
sets = {"ARC_0.0025": (W, dws), "ARC_0.001": (W, draws(150, 0.001, 101)), "ARC_0.005": (W, draws(150, 0.005, 102))}
for s in (106, 111, 115):
    sets[f"{s}_0.0025"] = (net(str(s)), draws(150, 0.0025, 200 + s))
for k, (Wk, d) in sets.items():
    a0 = accuracy(Wk, th.zeros(1, 432, device=DEV, dtype=DT))[0]
    a = accuracy(Wk, d)
    out[k + "_dw"] = d.cpu().numpy(); out[k + "_acc"] = a; out[k + "_clean"] = a0
    print(f"{k:12s} clean {100 * a0:.2f}%  copies: mean {100 * a.mean():.2f}, SD {100 * a.std():.2f}, range {100 * a.min():.1f}-{100 * a.max():.1f}  ({time.time() - t0:.0f} s)")
out["ARC_0.0025_acc100k"] = acc100k
np.savez("copies.npz", **out)

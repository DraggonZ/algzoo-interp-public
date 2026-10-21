"""Score the estimates on many networks. Usage: across.py set start end  (set: nets = ARC + 32 trained, order ARC,
100-115, 200-215; noisy = the 200 noisy copies of hilton_noise.py). Frozen settings (chosen on ARC and per-condition
checks on ARC and seed 208 only):
  old    : seven smaller numbers at their mean, 0/1 answer (the old grid estimate), split Gauss-Legendre grid n=32
  two    : two leads Gaussian through the Jacobian at the mean path, grid split-gl n=24
  final  : 'final' model of estimate.py (Q4=4, no screen, points below 1e-6 of the mass skipped), grid centred on
           the third (split3-gl), n=16. Frozen 2026-10-18 after development on ARC (and seed 208 per condition).
Appends: index old two final share-largest share-third share-another runs_old runs_two runs_final."""
import sys, time, numpy as np
from lib import *
from estimate import estimate
which, a, b = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
names = ["ARC"] + [str(s) for s in list(range(100, 116)) + list(range(200, 216))]
copies = noisy_copies() if which == "noisy" else None
if which == "shared":   # the coordinator's 100 copies at noise 0.0025 (truth on 2M lists each, ../truth/noisy_truth.npz)
    dw = np.load("../truth/noisy_truth.npz")["dw_0.0025"]; W0 = load_net("ARC"); copies = []
    for v in dw:
        v = torch.tensor(v, device=DEV, dtype=dt)
        copies.append((W0[0] + v[:16].reshape(W0[0].shape), W0[1] + v[16:272].reshape(16, 16), W0[2] + v[272:].reshape(10, 16)))
with open(f"across_{which}.txt", "a") as f:
    for i in range(a, b):
        W = load_net(names[i]) if which == "nets" else copies[i]
        t0 = time.time(); r0 = estimate(W, 32, 32, "mean01", kind="split-gl"); r2 = estimate(W, 24, 24, "two", kind="split-gl")
        r4 = estimate(W, 16, 16, "final", 4, kind="split3-gl")
        sp = r4["split"] / (1 - r4["acc"])
        f.write(f"{i} {r0['acc']:.6f} {r2['acc']:.6f} {r4['acc']:.6f} {sp[0]:.5f} {sp[1]:.5f} {sp[2]:.5f} {r0['runs']:.0f} {r2['runs']:.0f} {r4['runs']:.0f}\n"); f.flush()
        print(which, i, f"{100*r0['acc']:.3f} {100*r2['acc']:.3f} {100*r4['acc']:.3f}  {time.time()-t0:.0f}s", flush=True)

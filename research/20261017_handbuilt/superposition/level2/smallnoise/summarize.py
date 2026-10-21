"""Errors of the estimates against the truth, and the cost of sampling at the same error.
Truth: ARC + 32 trained networks from ../../../hilton/propagate/across.npy (10M-20M lists; column 0), the 200 noisy copies
from ../../../hilton/hilton_noise.npz ('true', 100k lists each), the 100 shared copies at 0.0025 from
../truth/noisy_truth.npz (2M lists each). Sampling at N lists has RMSE sqrt(mean p(1-p) / N)."""
import numpy as np
H = "/data/output/research/algzoo/20261017_handbuilt/hilton"
sets = {}
A = np.load(f"{H}/propagate/across.npy")
r = np.loadtxt("across_nets.txt"); r = r[np.argsort(r[:, 0])]
sets["ARC"] = (A[:1, 0], r[:1], A[:1, 1])
sets["32 trained networks (held out)"] = (A[1:, 0], r[1:], A[1:, 1])
try:
    r = np.loadtxt("across_noisy.txt", ndmin=2); r = r[np.argsort(r[:, 0])]; t = np.load(f"{H}/hilton_noise.npz")["true"][r[:, 0].astype(int)]
    sets[f"{len(r)} noisy copies (Hilton's noise, truth 100k lists)"] = (t, r, None)
except OSError: pass
try:
    r = np.loadtxt("across_shared.txt", ndmin=2); r = r[np.argsort(r[:, 0])]; t = np.load("../truth/noisy_truth.npz")["stats_0.0025"][r[:, 0].astype(int), 0]
    sets[f"{len(r)} shared noisy copies (noise 0.0025, truth 2M lists)"] = (t, r, None)
except OSError: pass
f = lambda e: np.sqrt(np.mean(e ** 2))
v2 = {}
for key, fn in [("ARC", "nets"), ("32", "nets"), ("noisy", "noisy"), ("shared", "shared")]:
    try:
        q = np.loadtxt(f"across_v2_{fn}.txt", ndmin=2); v2[key] = dict(zip(q[:, 0].astype(int), q))
    except OSError: pass
for name, (t, r, oldgrid) in sets.items():
    print(f"\n{name}: true accuracy {100*t.min():.2f}-{100*t.max():.2f}%")
    for k, col in [("old reduction (seven at mean, 0/1), new grid", 1), ("two leads Gaussian (Jacobian at mean path)", 2), ("final: fourth largest exact + Gaussian rest", 3)]:
        e = 100 * (r[:, col] - t)
        line = f"  {k:46s} RMSE {f(e):.2f}  mean error {e.mean():+.2f}  largest {np.abs(e).max():.2f}"
        if len(t) > 2: line += f"  correlation {np.corrcoef(r[:, col], t)[0, 1]:.3f}"
        print(line)
    if oldgrid is not None:
        e = 100 * (oldgrid - t); print(f"  {'old grid estimate as published (across.npy)':46s} RMSE {f(e):.2f}  mean error {e.mean():+.2f}  largest {np.abs(e).max():.2f}")
    key = "ARC" if name == "ARC" else "32" if name.startswith("32") else "noisy" if "Hilton" in name else "shared"
    off = 1 if key == "32" else 0
    if key in v2 and all(int(i) + 0 in v2[key] for i in r[:, 0]):
        q = np.array([v2[key][int(i)] for i in r[:, 0]]); e = 100 * (q[:, 1] - t)
        line = f"  {'v2 (revised after the frozen run)':46s} RMSE {f(e):.2f}  mean error {e.mean():+.2f}  largest {np.abs(e).max():.2f}"
        if len(t) > 2: line += f"  correlation {np.corrcoef(q[:, 1], t)[0, 1]:.3f}"
        print(line + f"  ({q[:,5].mean()/1e6:.0f}M run-equivalents; sampling at this RMSE needs about {np.mean(t*(1-t))/(f(e)/100)**2:,.0f} lists)")
        if key == "32":
            dev = np.isin(r[:, 0].astype(int), [15, 24])    # seeds 114 and 207 were used to choose v2
            e2 = e[~dev]; print(f"    v2 on the 30 networks not used for it: RMSE {f(e2):.2f}, mean error {e2.mean():+.2f}, largest {np.abs(e2).max():.2f}; old reduction on the same 30: RMSE {f(100*(r[~dev,1]-t[~dev])):.2f}")
            bad = np.argsort(-np.abs(e))[:4]; print("    largest v2 errors (row: truth, v2, old): " + "; ".join(f"{int(r[i,0])}: {100*t[i]:.2f}, {100*q[i,1]:.2f}, {100*r[i,1]:.2f}" for i in bad))
    if "Hilton" in name: print(f"  (truth standard error per copy about {100*np.sqrt(np.mean(t*(1-t))/1e5):.2f} points)")
    e3 = 100 * (r[:, 3] - t); e1 = 100 * (r[:, 1] - t)
    if len(t) > 1:
        print(f"  final minus old reduction: RMSE of difference {f(e3 - e1):.2f}; final better on {np.mean(np.abs(e3) < np.abs(e1)):.2f} of networks")
        bad = np.argsort(-np.abs(e3))[:3]; print("  largest final errors (row: truth, final, old): " + "; ".join(f"{int(r[i,0])}: {100*t[i]:.2f}, {100*r[i,3]:.2f}, {100*r[i,1]:.2f}" for i in bad))
    pq = np.mean(t * (1 - t)); runs = r[:, 9].mean()
    print(f"  cost: final {runs/1e6:.0f}M run-equivalents per network (old reduction {r[:,7].mean()/1e6:.1f}M, two {r[:,8].mean()/1e6:.0f}M); "
          f"sampling reaches the final RMSE {f(e3):.2f} with about {pq/(f(e3)/100)**2:,.0f} lists")

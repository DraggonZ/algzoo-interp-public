"""Development scores of the cheap tie-band estimator (cheap_est.py) on ARC and seeds 200-215, against the converged
reduction (propagate/across.npy) and the true accuracy (10M lists, propagate/across.npy)."""
import sys, numpy as np
import common as c, cheap_est as ce
log = c.Log("cheap_dev.log")
alln = [n for n, _ in c.nets()]; dev = ["ARC"] + [str(s) for s in c.SEEDS_DEV]
modes = sys.argv[1:] or ["double", "losers", "all"]
res = {m: [] for m in modes}
for name, W in c.nets():
    if name not in dev: continue
    i = alln.index(name); row = [f"{name:>4s}: true {100*c.TRUE[i]:.2f}, converged reduction {100*c.REF[i]:.2f}"]
    for m in modes:
        a, cost = ce.estimate(W, m); res[m].append((a, 100 * c.REF[i], 100 * c.TRUE[i], cost["runs"], cost["steps"]))
        row.append(f"{m} {a:.2f} ({cost['runs']} runs, {cost['steps']:.0f} step-equivalents)")
    log("; ".join(row))
for m in modes:
    r = np.array(res[m])
    log(f"mode {m}: mean cost {r[:,3].mean():.0f} runs ({r[:,4].mean():.0f} step-equivalents); RMSE against converged reduction {np.sqrt(((r[:,0]-r[:,1])**2).mean()):.2f}, "
        f"against truth {np.sqrt(((r[:,0]-r[:,2])**2).mean()):.2f} (mean error {np.mean(r[:,0]-r[:,2]):+.2f}); sampling at that many runs: {21/np.sqrt(r[:,3].mean()):.2f}")

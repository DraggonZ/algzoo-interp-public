"""Development scores of rootband.py (ARC and seeds 200-215) at several acceptance thresholds, against the converged
reduction and the truth. Step-equivalents: tie runs with shared prefixes, tangents at the tie counted from the first
moved position, search evaluations as full runs."""
import sys, numpy as np
import common as c, rootband as rb
log = c.Log("rootband_dev.log")
alln = [n for n, _ in c.nets()]; dev = ["ARC"] + [str(s) for s in c.SEEDS_DEV]
accs = [float(x) for x in sys.argv[1:]] or [0.05, 0.1, 0.3]
res = {a: [] for a in accs}
for name, W in c.nets():
    if name not in dev: continue
    i = alln.index(name); row = [f"{name:>4s}: true {100*c.TRUE[i]:.2f}, converged {100*c.REF[i]:.2f}"]
    for ac in accs:
        a, cost = rb.estimate(W, accept=ac); res[ac].append((a, 100 * c.REF[i], 100 * c.TRUE[i], cost["runs"], cost["steps"]))
        row.append(f"accept {ac}: {a:.2f} ({cost['runs']} runs, {cost['steps']:.0f} step-eq, searched {cost['search23']}+{cost['search12']})")
    log("; ".join(row))
for ac in accs:
    r = np.array(res[ac])
    log(f"accept {ac}: mean cost {r[:,3].mean():.0f} runs ({r[:,4].mean():.0f} step-equivalents); RMSE against converged {np.sqrt(((r[:,0]-r[:,1])**2).mean()):.2f}, "
        f"against truth {np.sqrt(((r[:,0]-r[:,2])**2).mean()):.2f} (mean error {np.mean(r[:,0]-r[:,2]):+.2f}); sampling: {21/np.sqrt(r[:,3].mean()):.2f} at the same runs, {21/np.sqrt(r[:,4].mean()):.2f} at the same step-equivalents")

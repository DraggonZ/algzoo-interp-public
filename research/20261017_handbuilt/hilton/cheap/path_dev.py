"""Development scores of pathband.py on ARC and seeds 200-215: exact widths (up to 60 linear pieces per band) and
cheaper versions that stop after 1, 2 or 3 pieces and extrapolate the last piece's straight line. Against the
converged reduction (used to choose) and the truth (reported only). Cost: passes (tie forward runs + tangent passes,
each counted as one run) and RNN-step equivalents (shared prefixes, tangents from the first moved step)."""
import sys, numpy as np
import common as c, pathband as pb
log = c.Log("path_dev.log")
alln = [n for n, _ in c.nets()]; dev = ["ARC"] + [str(s) for s in c.SEEDS_DEV]
Ks = [int(x) for x in sys.argv[1:]] or [1, 2, 3, 60]
res = {k: [] for k in Ks}
for name, W in c.nets():
    if name not in dev: continue
    i = alln.index(name); row = [f"{name:>4s}: true {100*c.TRUE[i]:.2f}, converged {100*c.REF[i]:.2f}"]
    for k in Ks:
        a, cost = pb.estimate(W, maxp=k); res[k].append((a, 100 * c.REF[i], 100 * c.TRUE[i], cost["runs"], cost["steps"]))
        row.append(f"{k} pieces: {a:.2f} ({cost['runs']} runs, {cost['steps']:.0f} step-eq)")
    log("; ".join(row))
for k in Ks:
    r = np.array(res[k])
    log(f"up to {k} pieces: mean cost {r[:,3].mean():.0f} runs ({r[:,4].mean():.0f} step-equivalents); error against converged reduction: RMSE {np.sqrt(((r[:,0]-r[:,1])**2).mean()):.2f}, mean {np.mean(r[:,0]-r[:,1]):+.2f}; "
        f"against truth: RMSE {np.sqrt(((r[:,0]-r[:,2])**2).mean()):.2f}, mean {np.mean(r[:,0]-r[:,2]):+.2f}; sampling at the same runs {21/np.sqrt(r[:,3].mean()):.2f}, at the same step-equivalents {21/np.sqrt(r[:,4].mean()):.2f}")
r = np.array(res[Ks[0]]); log(f"converged reduction against truth on these networks: RMSE {np.sqrt(((r[:,1]-r[:,2])**2).mean()):.2f}")

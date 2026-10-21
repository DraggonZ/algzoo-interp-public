"""Development: the tie-band estimator with slopes from forward tangents at the tie (cheap_est.py, slope_rule
'tangent'): (a) every loser's own tangent (falls back to the median where its tangent is <= 0); (b) one median slope
per tie type from n_tan tangents at evenly spaced losers. Bulk runs as in mode 'double' (only triples wrong at both
ties; a bulk margin <= 0 marks a wide band). Against converged reduction and truth on ARC and seeds 200-215."""
import sys, numpy as np
import common as c, cheap_est as ce
log = c.Log("cheap_dev2.log")
alln = [n for n, _ in c.nets()]; dev = ["ARC"] + [str(s) for s in c.SEEDS_DEV]
variants = [("own tangents", dict(slope_rule="tangent")), ("median of 36 tangents", dict(slope_rule="tangent", n_tan=36)),
            ("median of 12 tangents", dict(slope_rule="tangent", n_tan=12))]
res = {v: [] for v, _ in variants}
for name, W in c.nets():
    if name not in dev: continue
    i = alln.index(name); row = [f"{name:>4s}: true {100*c.TRUE[i]:.2f}, converged {100*c.REF[i]:.2f}"]
    for v, kw in variants:
        a, cost = ce.estimate(W, "double", **kw); res[v].append((a, 100 * c.REF[i], 100 * c.TRUE[i], cost["runs"], cost["steps"]))
        row.append(f"{v} {a:.2f} ({cost['runs']} runs)")
    log("; ".join(row))
for v, _ in variants:
    r = np.array(res[v])
    log(f"{v}: mean cost {r[:,3].mean():.0f} runs ({r[:,4].mean():.0f} step-equivalents, tangents as full runs); RMSE against converged {np.sqrt(((r[:,0]-r[:,1])**2).mean()):.2f}, "
        f"against truth {np.sqrt(((r[:,0]-r[:,2])**2).mean()):.2f} (mean error {np.mean(r[:,0]-r[:,2]):+.2f}); sampling at that many runs {21/np.sqrt(r[:,3].mean()):.2f}")

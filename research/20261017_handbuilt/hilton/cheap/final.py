"""Frozen method, scored once: tie runs at two nodes, exact band widths by following linear pieces (pathband.py, up to
60 pieces), band model integrated by the network-independent table (table.py). Development networks: ARC and seeds
200-215; held out: seeds 100-115 (never looked at before this run). Truth: 10M random lists each
(propagate/across.npy). Sampling's RMSE at N lists: sqrt(mean over the networks of p(1 - p) / N)."""
import numpy as np, time
import common as c, pathband as pb, table
log = c.Log("final.log")
G = np.load("table.npz")["G"]
alln = [n for n, _ in c.nets()]; rows = {}
t0 = time.time()
for name, W in c.nets():
    _, cost, w = pb.estimate(W, return_parts=True)
    e = table.accuracy(w["23"], w["12"], G); i = alln.index(name)
    rows[name] = (e, 100 * c.TRUE[i], 100 * c.REF[i], cost["runs"], cost["steps"])
    log(f"{name:>4s}: estimate {e:.2f}%  true {100*c.TRUE[i]:.2f}%  error {e-100*c.TRUE[i]:+.2f}  (converged reduction {100*c.REF[i]:.2f}); cost {cost['runs']} runs, {cost['steps']:.0f} step-equivalents")
log(f"(GPU time for all 33 networks {time.time()-t0:.0f}s)")
for grp, names in (("development (ARC, 200-215)", ["ARC"] + [str(s) for s in c.SEEDS_DEV]), ("held out (100-115)", [str(s) for s in c.SEEDS_HELD]), ("all 33", alln)):
    r = np.array([rows[n] for n in names]); p = r[:, 1] / 100
    N = r[:, 3].mean(); Ns = r[:, 4].mean(); sd1 = 100 * np.sqrt((p * (1 - p)).mean())
    log(f"{grp}: RMSE {np.sqrt(((r[:,0]-r[:,1])**2).mean()):.2f} points, mean error {np.mean(r[:,0]-r[:,1]):+.2f}, largest {np.abs(r[:,0]-r[:,1]).max():.2f}; "
        f"against the converged reduction {np.sqrt(((r[:,0]-r[:,2])**2).mean()):.2f}; converged reduction against truth {np.sqrt(((r[:,2]-r[:,1])**2).mean()):.2f}. "
        f"Mean cost {N:.0f} runs ({r[:,3].min():.0f}-{r[:,3].max():.0f}), {Ns:.0f} step-equivalents. Sampling RMSE at {N:.0f} lists {sd1/np.sqrt(N):.2f}, at {Ns:.0f} lists {sd1/np.sqrt(Ns):.2f}; "
        f"lists sampling needs for our RMSE: {(sd1/np.sqrt(((r[:,0]-r[:,1])**2).mean()))**2:.0f}")

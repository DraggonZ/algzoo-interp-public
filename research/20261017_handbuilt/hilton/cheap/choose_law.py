"""Development choice of the scale law (how a node's band width is carried to other u and gap), made against the
converged reduction only (not the truth): exact path-following widths at the nodes (pathband.py), four laws, on ARC
and seeds 200-215. Truth RMSE printed for information, not used to choose."""
import numpy as np, torch
import common as c, bandlib as bl, pathband as pb
log = c.Log("choose_law.log")
alln = [n for n, _ in c.nets()]; dev = ["ARC"] + [str(s) for s in c.SEEDS_DEV]
laws = ["const", "norm", "proj", "spread"]; res = []
for name, W in c.nets():
    if name not in dev: continue
    _, _, w = pb.estimate(W, return_parts=True); np.savez(f"path_widths_{name}.npz", A=w["23"], B=w["12"])
    tt = lambda x: torch.tensor(x, device=c.DEV, dtype=c.dt)
    i = alln.index(name); row = [bl.accuracy(tt(w["23"]), tt(w["12"]), law) for law in laws]
    res.append(row + [100 * c.REF[i], 100 * c.TRUE[i]])
    log(f"{name:>4s}: converged {100*c.REF[i]:.2f}; " + ", ".join(f"{l} {v:.2f}" for l, v in zip(laws, row)))
r = np.array(res)
for j, l in enumerate(laws):
    log(f"law {l}: RMSE against converged reduction {np.sqrt(((r[:,j]-r[:,4])**2).mean()):.2f} (mean {np.mean(r[:,j]-r[:,4]):+.2f}); against truth {np.sqrt(((r[:,j]-r[:,5])**2).mean()):.2f}")

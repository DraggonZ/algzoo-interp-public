"""Score the weights-only per-condition models (models.py) against the sampled truth in conds_ARC.npz."""
import sys, numpy as np, torch
from lib import *
import models
name = sys.argv[1] if len(sys.argv) > 1 else "ARC"; W = load_net(name); z = np.load(f"conds_{name}.npz")
for label in ["random", "near"]:
    pos = torch.tensor(z[label + "_pos"], device=DEV); val = torch.tensor(z[label + "_val"], device=DEV, dtype=dt); tr = z[label + "_truth"]
    X0, _ = base_inputs(pos, val); m01 = (run(W, X0).argmax(1) == pos[:, 1]).double().cpu().numpy()
    out = {"mean 0/1 (old grid)": (m01, None),
           "two": [t.cpu().numpy() for t in models.model_two(W, pos, val)],
           "two+1d": [t.cpu().numpy() for t in models.model_two(W, pos, val, True)],
           "fourth Q4=8": [t.cpu().numpy() for t in models.model_fourth(W, pos, val, 8)],
           "fourth Q4=16": [t.cpu().numpy() for t in models.model_fourth(W, pos, val, 16)]}
    print(f"{label} conditions: truth P(right) {tr[:,0].mean():.4f}; wrong to largest {tr[:,1].mean():.4f}, third {tr[:,2].mean():.4f}, smaller {tr[:,3].mean():.4f}")
    for k, (p, s) in out.items():
        line = f"  {k:20s} P(right) {p.mean():.4f} bias {p.mean()-tr[:,0].mean():+.4f} mean abs error {np.abs(p-tr[:,0]).mean():.4f} corr {np.corrcoef(p, tr[:,0])[0,1]:.3f}"
        if s is not None: line += f" | wrong to largest {s[:,0].mean():.4f}, third {s[:,1].mean():.4f}, smaller {s[:,2].mean():.4f}"
        print(line)

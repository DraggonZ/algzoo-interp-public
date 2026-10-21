"""Per-condition check by arrival order (does the third arrive after the largest?): is the order-specific error in
'third wins' from the per-condition model or from the grid? Truth from conds_ARC.npz (sampled, diagnosis only)."""
import numpy as np, torch, models
from lib import *
W = load_net("ARC"); z = np.load("conds_ARC.npz")
for label in ["random", "near"]:
    o = z[label + "_pos"]; pos = torch.tensor(o, device=DEV); val = torch.tensor(z[label + "_val"], device=DEV, dtype=dt); tr = z[label + "_truth"]
    p, sp = (t.cpu().numpy() for t in models.model_fourth(W, pos, val, 8))
    X0, _ = base_inputs(pos, val); a = run(W, X0).argmax(1).cpu().numpy(); old3 = (a == o[:, 2]).astype(float)
    for nm, m in [("third after the largest", o[:, 2] > o[:, 0]), ("third before the largest", o[:, 2] < o[:, 0])]:
        print(f"{label:6s} {nm:25s} ({m.sum()}): third wins truth {tr[m,2].mean():.4f}, fourth model {sp[m,1].mean():.4f}, old 0/1 {old3[m].mean():.4f}; "
              f"P(right) truth {tr[m,0].mean():.4f} model {p[m].mean():.4f}")

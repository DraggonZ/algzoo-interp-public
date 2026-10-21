"""Same nine-lead variant on the random and near-tie conditions (mostly third > 0), split by the third's sign."""
import numpy as np, torch, models
from lib import *
for net in ["ARC", "208"]:
    W = load_net(net); z = np.load(f"conds_{net}.npz")
    for label in ["random", "near"]:
        pos = torch.tensor(z[label + "_pos"], device=DEV); val = torch.tensor(z[label + "_val"], device=DEV, dtype=dt); tr = z[label + "_truth"]
        p3 = models.model_fourth(W, pos, val, 8)[0].cpu().numpy(); p9 = models.model_fourth(W, pos, val, 8, True)[0].cpu().numpy()
        for sg, m in [("third > 0", z[label + "_val"][:, 2] > 0), ("third < 0", z[label + "_val"][:, 2] < 0)]:
            if m.sum() < 5: continue
            t = tr[m, 0]
            print(f"{net} {label:6s} {sg} ({m.sum()} conditions): truth {t.mean():.4f}; 3 leads bias {p3[m].mean()-t.mean():+.4f} MAE {np.abs(p3[m]-t).mean():.4f}; 9 leads bias {p9[m].mean()-t.mean():+.4f} MAE {np.abs(p9[m]-t).mean():.4f}")

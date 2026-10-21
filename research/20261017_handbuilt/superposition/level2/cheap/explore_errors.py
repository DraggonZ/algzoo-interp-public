"""Where ARC's mistakes sit: by gaps between the three largest and by the second-largest's sign."""
import sys, numpy as np, torch
sys.path.insert(0, "../../../family")
from common import *
torch.manual_seed(0)
W = ARC; n = 0; rows = []
for it in range(10):
    X = torch.randn(2_000_000, 10, device=DEV, dtype=dt)
    a = scores(W, X).argmax(1); v, o = X.sort(1, descending=True)
    wr = a != o[:, 1]
    kind = torch.where(a == o[:, 0], 0, torch.where(a == o[:, 2], 1, 2))
    rows.append(torch.stack([wr.double(), kind.double(), v[:, 0], v[:, 1], v[:, 2]], 1).cpu().numpy())
R = np.concatenate(rows); wr = R[:, 0] > 0; k = R[:, 1]; L, S, T = R[:, 2], R[:, 3], R[:, 4]
print("accuracy", 1 - wr.mean())
for name, m in (("largest", k == 0), ("third", k == 1), ("other", k == 2)):
    mm = wr & m; print(f"{name}: {100*mm.mean():.3f}% of lists; S<0 share {np.mean(S[mm] < 0):.2f}")
    g = (S - T)[mm] if name != "largest" else (L - S)[mm]
    print("   gap quantiles", np.round(np.quantile(g, [.1, .25, .5, .75, .9, .99]), 3), " rel gap", np.round(np.quantile(g / np.abs(S[mm]), [.5, .9, .99]), 3))
print("P(S<0)", np.mean(S < 0), "acc given S<0", 1 - wr[S < 0].mean())

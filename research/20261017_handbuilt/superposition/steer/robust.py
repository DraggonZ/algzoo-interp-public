"""Step 5: do the account-designed edits keep working on noisy copies of ARC (weight noise 0.0025)? Prediction: an edit
resting on the mechanism shifts each copy by about the same amount as ARC. Edits: (A) the hiding neurons' last-step
input x0.99 (ARC: last-step contender's near-tie wins 57.5% -> 50.0%, accuracy unchanged); (B) store x1.01 at step 5
(ARC: earlier contender's near-tie wins up about 8-11 points); (C) memory x1.05 at step 5 alone (ARC: earlier contender
down about 17 points)."""
import numpy as np
from harness import *
d = np.load("../level2/truth/noisy_truth.npz")
def copies(k=20):
    shapes = [p.shape for p in ARC]; sizes = [p.numel() for p in ARC]
    for v in d["dw_0.0025"][:k]:
        out, i = [], 0
        for p, sz in zip(ARC, sizes): out.append(p + torch.tensor(v[i:i + sz], device=DEV, dtype=dt).reshape(p.shape)); i += sz
        yield out
def last_hiding(W, X, b=0.01):
    f = torch.ones(16, device=DEV, dtype=dt); f[[1, 6, 7, 8]] = 1 - b
    h = torch.zeros(len(X), 16, device=DEV, dtype=dt)
    for t in range(10): h = torch.relu(h @ W[1].T + X[:, t:t + 1] * W[0] * (f if t == 9 else 1))
    return (h @ W[2].T).argmax(1)
X = lists(2_000_000, seed=31); o = X.argsort(1, descending=True); s = o[:, 1]; v = X.sort(1, descending=True).values
near = (v[:, 1] > 0) & ((v[:, 1] - v[:, 2]) < 0.05 * v[:, 1]); late9 = near & ((o[:, 1] == 9) | (o[:, 2] == 9))
l = o[:, 0]; t3 = o[:, 2]; early = torch.where(s < t3, s, t3); lat = torch.where(s < t3, t3, s)
mid = near & (l <= 3) & (early > l) & (early <= 3) & (lat >= 6)
rows = []
for name, W in [("ARC", ARC)] + [(f"copy {i}", W) for i, W in enumerate(copies())]:
    a0 = run(W, X).argmax(1); aA = last_hiding(W, X)
    aB = run(W, X, [([2, 4], {5}, 1.01)]).argmax(1); aC = run(W, X, [(GROUPS["memory"], {5}, 1.05)]).argmax(1)
    r = [100 * (a0 == s).double().mean().item(), 100 * (aA == s).double().mean().item(),
         100 * (a0[late9] == 9).double().mean().item(), 100 * (aA[late9] == 9).double().mean().item(),
         100 * (a0[mid] == early[mid]).double().mean().item(), 100 * (aB[mid] == early[mid]).double().mean().item(), 100 * (aC[mid] == early[mid]).double().mean().item()]
    rows.append(r)
    print(f"{name:8s} acc {r[0]:.2f} -> {r[1]:.2f} | last-step wins {r[2]:5.1f} -> {r[3]:5.1f} | earlier contender wins {r[4]:5.1f}: store up {r[5]:5.1f}, memory up {r[6]:5.1f}", flush=True)
R = np.array(rows[1:])
print(f"copies: accuracy change {np.mean(R[:,1]-R[:,0]):+.3f} ± {np.std(R[:,1]-R[:,0]):.3f}; last-step bias change {np.mean(R[:,3]-R[:,2]):+.1f} ± {np.std(R[:,3]-R[:,2]):.1f} (ARC {rows[0][3]-rows[0][2]:+.1f}); store edit {np.mean(R[:,5]-R[:,4]):+.1f} ± {np.std(R[:,5]-R[:,4]):.1f} (ARC {rows[0][5]-rows[0][4]:+.1f}); memory edit {np.mean(R[:,6]-R[:,4]):+.1f} ± {np.std(R[:,6]-R[:,4]):.1f} (ARC {rows[0][6]-rows[0][4]:+.1f})")

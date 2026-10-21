"""What does ARC's neuron 8 hold? Its pre-activation at each step, regressed (least squares, analysis only) on list
quantities: the current number x_t, the largest before it m1, the runner-up before it m2, the jump of the step before,
and, as wave proxies, the other hiding neurons' previous values. Shares of variance explained, per set of regressors,
held out. Steps 3-9 pooled."""
import sys; sys.path.insert(0, "/data/output/research/algzoo/20261017_handbuilt/family")
import numpy as np, torch
from common import DEV, dt, ARC
g = torch.Generator(device=DEV).manual_seed(5); X = torch.randn(200_000, 10, device=DEV, dtype=dt, generator=g)
W = ARC; h = torch.zeros(len(X), 16, device=DEV, dtype=dt); Z, H = [], [h]
for t in range(10):
    z = h @ W[1].T + X[:, t:t + 1] * W[0]; h = torch.relu(z); Z.append(z); H.append(h)
Xn = X.cpu().numpy(); m1 = np.zeros(len(Xn)); m2 = np.zeros(len(Xn)); rows = {k: [] for k in ("x", "m1", "m2", "d", "hid", "y")}
for t in range(10):
    x = Xn[:, t]; d = np.maximum(x - m1, 0)
    if t >= 3:
        rows["x"].append(x); rows["m1"].append(m1.copy()); rows["m2"].append(m2.copy())
        rows["hid"].append(H[t][:, [1, 6, 7, 8]].cpu().numpy()); rows["y"].append(Z[t][:, 8].cpu().numpy())
        rows["d"].append(np.maximum(Xn[:, t - 1] - np.maximum.accumulate(np.c_[np.zeros(len(Xn)), Xn[:, :t - 1]], 1)[:, -1], 0))
    m2 = np.maximum(m2, np.minimum(x, m1)); m1 = np.maximum(m1, x)
R = {k: np.concatenate(v) for k, v in rows.items()}
def r2(cols):
    A = np.column_stack([R[c] for c in cols])
    n = len(A) // 2; coef = np.linalg.lstsq(A[:n], R["y"][:n], rcond=None)[0]; res = R["y"][n:] - A[n:] @ coef
    return 1 - res.var() / R["y"][n:].var(), coef
for cols in (["x", "m1"], ["x", "m2"], ["x", "m1", "m2"], ["x", "m1", "m2", "d"], ["x", "m1", "m2", "d", "hid"], ["x", "m1", "hid"]):
    v, c = r2(cols); print(f"{' + '.join(cols):24s} R2 {v:.3f}  coefficients " + " ".join(f"{a:+.2f}" for a in c))
print(f"neuron 8 on (pre-activation > 0) on {100*(R['y'] > 0).mean():.0f}% of steps 3-9")

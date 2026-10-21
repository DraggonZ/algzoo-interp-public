"""Review 5 check E: k-spike accuracy Z_k (keep the k largest inputs, others set to a background value, target the
original 2nd position) on the six trained models, against full accuracy. Sampled, diagnostic only."""
import numpy as np
from algzoo.core import example, zoo
rng = np.random.default_rng(21); N = 1 << 21
X = rng.standard_normal((N, 10)); o = np.argsort(X, 1); p2 = o[:, -2]
def acc(params, Xin):
    w_hi, w_hh, w_oh = (np.asarray(p, np.float64) for p in params)
    h = np.zeros((len(Xin), 16))
    for t in range(10):
        h = np.maximum(0, h @ w_hh.T + Xin[:, t, None] * w_hi)
    return ((h @ w_oh.T).argmax(1) == p2).mean()
rows = []
for name in ["example", 0, 1, 2, 3, 4]:
    params = example() if name == "example" else zoo(16, 10, name)
    full = acc(params, X); out = [full]
    for k in [2, 3, 4]:
        for bg in [0.0, -0.5]:
            Xk = np.full_like(X, bg); idx = o[:, -k:]
            np.put_along_axis(Xk, idx, np.take_along_axis(X, idx, 1), 1)
            out.append(acc(params, Xk))
    rows.append(out)
    print(name, " ".join(f"{v:.3f}" for v in out), flush=True)
R = np.array(rows); err = R[:, 1:] - R[:, :1]
print("columns: full, Z2 bg0, Z2 bg-0.5, Z3 bg0, Z3 bg-0.5, Z4 bg0, Z4 bg-0.5")
print("RMSE vs full over six (points):", " ".join(f"{100*np.sqrt(np.mean(e**2)):.1f}" for e in err.T))

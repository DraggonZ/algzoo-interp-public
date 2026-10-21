"""Main components of the near-tie lean. A small logistic model with named terms (the ratio third/second, the
late-number bias, which contender comes later, closeness to the largest, the largest's size) against the trees
(AUC 0.90 from positions and two ratios). Also: neuron 7's final contribution to the lead alone, as a predictor."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from flips import *
N = 400000; Xs = X[:N]; nn = torch.arange(N, device=DEV); Ls, Ss, Ts = L[:N], S[:N], T[:N]
y = (d0[:N] == Ts).cpu().numpy(); xs = Xs[nn, Ss]
r = (Xs[nn, Ts] / xs).cpu().numpy(); rl = (Xs[nn, Ls] / xs).cpu().numpy()
f = lambda t: t.cpu().numpy().astype(float)
feats = {"gap: 1 - third/second": 1 - r, "third arrives later": f(Ts > Ss), "third at step 9": f(Ts == 9), "second at step 9": f(Ss == 9),
         "third right after largest": f(Ts == Ls + 1), "second right after largest": f(Ss == Ls + 1),
         "largest before both": f(Ls < torch.minimum(Ss, Ts)), "largest between": f((Ls > torch.minimum(Ss, Ts)) & (Ls < torch.maximum(Ss, Ts))),
         "distance between contenders": f((Ss - Ts).abs()), "largest/second": rl,
         "third's position": f(Ts), "second's position": f(Ss)}
F = np.c_[[v for v in feats.values()]].T
Fz = (F - F[:300000].mean(0)) / F[:300000].std(0)
lr = LogisticRegression(max_iter=2000).fit(Fz[:300000], y[:300000]); p = lr.predict_proba(Fz[300000:])[:, 1]
print(f"logistic, {F.shape[1]} named terms: held-out AUC {roc_auc_score(y[300000:], p):.3f}, accuracy {100*((p > 0.5) == y[300000:]).mean():.1f}% (trees 0.897 / 86.1%, base 79.3%)")
for (k, _), c in sorted(zip(feats.items(), lr.coef_[0]), key=lambda z: -abs(z[1])): print(f"   {k:30s} {c:+.2f}")
for k in ("gap: 1 - third/second", "third at step 9"):
    a = roc_auc_score(y, -feats[k] if k.startswith("gap") else feats[k]); print(f"   {k} alone: AUC {a:.3f}")
# neuron 7's final contribution to the lead
_, Hs = run(ARC, Xs, keep=True); h = Hs[-1]; c7 = (h[:, 7] * (ARC[2][Ts, 7] - ARC[2][Ss, 7])).cpu().numpy()
print(f"neuron 7's final push toward the third, alone: AUC {roc_auc_score(y, c7):.3f}")

"""How much of the near-tie decision do the account's variables carry? Predict 'the network names the third' on near
ties from feature sets of increasing detail, with gradient-boosted trees (an analysis fit on the trained network's
outputs; held-out lists). F1 positions of the largest, second, third. F2 + value ratios third/second, largest/second.
F3 + count and positions of smaller numbers above 0.75 of the second (the detector keeps those). F4 all ten values
divided by the second, in order (everything the network sees)."""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from flips import *
N = 400000; Xs = X[:N]; y = (d0[:N] == T[:N]).cpu().numpy(); nn = torch.arange(N, device=DEV)
xs = Xs[nn, S[:N]]
pos = torch.stack([L[:N], S[:N], T[:N]], 1).cpu().numpy().astype(float)
rat = torch.stack([Xs[nn, T[:N]] / xs, Xs[nn, L[:N]] / xs], 1).cpu().numpy()
kept = (Xs > 0.75 * xs[:, None]) & small[:N]
F3 = np.c_[pos, rat, kept.sum(1).cpu().numpy(), kept.cpu().numpy().astype(float)]
F4 = np.c_[pos, (Xs / xs[:, None]).cpu().numpy()]
for lab, F in (("F1 positions", pos), ("F2 + ratios", np.c_[pos, rat]), ("F3 + kept smaller numbers", F3), ("F4 all values", F4)):
    clf = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.1, max_leaf_nodes=63).fit(F[:300000], y[:300000])
    p = clf.predict_proba(F[300000:])[:, 1]
    print(f"{lab:28s}: held-out accuracy {100*((p > 0.5) == y[300000:]).mean():.1f}% (always 'second' {100*(1-y[300000:].mean()):.1f}%), AUC {roc_auc_score(y[300000:], p):.3f}", flush=True)

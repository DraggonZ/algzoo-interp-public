"""content_rules.py plus every past number and its positive part by lag (steps ago, 0-9), for neurons 3, 12, 15: does
the memory's vote content set their switching? Also a depth-8 tree on all features, as a ceiling for simple rules."""
import os; os.environ.setdefault("OMP_NUM_THREADS", "4")
import sys, numpy as np, torch
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
sys.path.insert(0, "../../family")
from common import *
from patterns import run_patterns
from content_rules import content
def lags(X):
    F = []
    for t in range(10):
        L = torch.zeros(len(X), 10, device=DEV, dtype=dt)
        for k in range(t + 1): L[:, k] = X[:, t - k]
        F.append(torch.cat([L, L.clamp(min=0)], 1))
    return torch.stack(F, 1)
if __name__ == "__main__":
    X = torch.randn(200000, 10, device=DEV, dtype=dt, generator=torch.Generator(device=DEV).manual_seed(103))
    P = run_patterns(ARC, X)[:, 1:]; F = torch.cat([content(X), lags(X)], 2)[:, 1:]; F = F.reshape(-1, F.shape[2]).cpu().numpy(); half = len(F) // 2
    for j in (3, 12, 15):
        y = P[..., j].reshape(-1).cpu().numpy(); base = max(y[half:].mean(), 1 - y[half:].mean())
        lr = LogisticRegression(C=10.0, max_iter=3000).fit(F[:half], y[:half]); acc = (lr.predict(F[half:]) == y[half:]).mean()
        gb = HistGradientBoostingClassifier(max_iter=200, max_depth=8).fit(F[:half], y[:half]); acc2 = (gb.predict(F[half:]) == y[half:]).mean()
        print(f"neuron {j:2d}: guess {100*base:5.1f}%, linear threshold on content + lags {100*acc:5.1f}%, boosted trees {100*acc2:5.1f}%")

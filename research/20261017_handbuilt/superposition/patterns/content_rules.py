"""Neurons 3, 12, 15 (and the others, for comparison): is each one's on/off a single linear threshold on list content?
Features: the largest so far, the previous largest and the second-largest so far, each times a one-hot of the age of the
largest (0-9), plus the current and previous number and the step one-hot (about 50 numbers). Logistic regression,
no intercept beyond the step terms; held-out accuracy against the depth-5 tree (switch_rules.log) and against guessing."""
import os; os.environ.setdefault("OMP_NUM_THREADS", "4")
import sys, numpy as np, torch
from sklearn.linear_model import LogisticRegression
sys.path.insert(0, "../../family")
from common import *
from patterns import run_patterns
def content(X):
    n = len(X); m1 = torch.zeros(n, device=DEV, dtype=dt); m2 = m1.clone(); Pv = m1.clone(); Lp = torch.full((n,), -1, device=DEV, dtype=torch.long)
    xprev = m1.clone(); F = []
    for t in range(10):
        x = X[:, t]; rec = x > m1
        Pv = torch.where(rec, m1, Pv); m2 = torch.where(rec, m1, torch.maximum(m2, x)); Lp = torch.where(rec, t, Lp); m1 = torch.maximum(m1, x)
        age = torch.where(Lp >= 0, t - Lp, 0); oh = torch.nn.functional.one_hot(age, 10).to(dt)
        st = torch.nn.functional.one_hot(torch.full((n,), t, device=DEV), 10).to(dt)
        F.append(torch.cat([oh * m1[:, None], oh * Pv[:, None], oh * m2[:, None], x[:, None], xprev[:, None], st], 1)); xprev = x
    return torch.stack(F, 1)
if __name__ == "__main__":
    X = torch.randn(200000, 10, device=DEV, dtype=dt, generator=torch.Generator(device=DEV).manual_seed(103))
    P = run_patterns(ARC, X)[:, 1:]; F = content(X)[:, 1:].reshape(-1, 42).cpu().numpy(); half = len(F) // 2
    for j in (3, 12, 15, 0, 10, 11, 13, 14):
        y = P[..., j].reshape(-1).cpu().numpy(); base = max(y[half:].mean(), 1 - y[half:].mean())
        lr = LogisticRegression(C=10.0, max_iter=2000).fit(F[:half], y[:half]); acc = (lr.predict(F[half:]) == y[half:]).mean()
        print(f"neuron {j:2d}: on {100*y.mean():5.1f}%, guess {100*base:5.1f}%, linear threshold on content {100*acc:5.1f}%")

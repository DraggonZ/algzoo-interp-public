"""Copy of ../memory/sign_rules3_fresh.py on this study's lists (test seed 31, fit seed 32), closed loop only.
Question (follows sign_rules2.py, 92.8%): do rotating-wave features of the earlier records and of the second-largest
make the switching rules good enough for under 2 points? Same as sign_rules2.py, with extra features in each
(step, age) cell: the previous largest p and the second-largest so far m2, each times cos and sin of
(40, 80, 125, 180 degrees x steps since it arrived). Fitted on 200,000 lists (seed 12), tested on 200,000 fresh lists (seed 99, not used while choosing the features).
(Analysis fit on the trained network; no hand-built weight comes from it.)"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "memory"))
import numpy as np
import sign_rules2 as s2
from memlib import run, task, lists, score, kinds, fmt

_base_feats = s2.feats


def ages(X):
    """position of the previous largest (before the current one) and of the second-largest so far, per step"""
    n, T = X.shape; pp = np.full((n, T), -1); p2 = np.full((n, T), -1)
    m1 = np.zeros(n); a1 = np.full(n, -1); prev = np.full(n, -1); m2 = np.zeros(n); a2 = np.full(n, -1)
    for t in range(T):
        x = X[:, t]; new = x > m1
        prev = np.where(new, a1, prev); a2 = np.where(new, a1, np.where(x > m2, t, a2))
        m2 = np.where(new, m1, np.maximum(m2, x)); a1 = np.where(new, t, a1); m1 = np.where(new, x, m1)
        pp[:, t] = prev; p2[:, t] = a2
    return pp, p2


def feats(X, V, H, t):
    F = _base_feats(X, V, H, t); pp, p2 = V["_pp"], V["_p2"]; extra = []
    for val, pos in ((V["m1prev"][:, t] * 0 + np.where(pp[:, t] >= 0, X[np.arange(len(X)), np.maximum(pp[:, t], 0)], 0), pp[:, t]), (V["m2"][:, t], p2[:, t])):
        age = np.where(pos >= 0, t - pos, 0)
        for th in (40, 80, 125, 180):
            extra += [val * np.cos(np.radians(th) * age), val * np.sin(np.radians(th) * age)]
    return np.c_[F, np.stack(extra, 1)]


s2.feats = feats
_task = s2.task
def task2(X):
    V = _task(X); V["_pp"], V["_p2"] = ages(X); return V
s2.task = task2

X = lists(31, 200_000); K = kinds(X); V = task2(X)
S0, P0, H0 = run(X); base = score(X, S0, S0, K)
R = s2.build_rules(lists(32, 200_000))
off = s2.apply_rules(R, X, V, H0); true_off = P0[:, :, s2.MEM] <= 0
print("on/off agreement with the true sign (test lists): " + " ".join(f"{n}:{np.mean(off[:, :, j] == true_off[:, :, j]):.3f}" for j, n in enumerate(s2.MEM)))
for mode in ["closed"]:
    S = s2.machine(X, H0, off, mode); r = score(X, S0, S, K)
    print(f"== all eight by the rule, {mode}: acc {100 * r['all'][0]:.1f} ({100 * (r['all'][0] - base['all'][0]):+.1f}) agree {100 * r['all'][1]:.1f} KL {r['all'][2]:.3f}")
    print(fmt(r, base))

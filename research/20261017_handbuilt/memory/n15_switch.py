"""Question: what decides neuron 15's on/off? (1) On lists with one positive number only (a lone record), its
pre-activation per unit of that number, by step and age: does a lone record's wave alone switch it? (2) In general
lists, a compact rule: in each age class, a linear fit of the pre-activation on a few named quantities (largest so
far m1, the arriving number x and max(x, 0), the second-largest so far m2, the previous largest p, step t times m1,
step t times m2); on/off agreement and the accuracy when 15 follows the rule (all other neurons by their own sign),
against the rich per-(step, age) rule of sign_rules3.py (agreement 0.944). Fitted on 200,000 lists (seed 12), tested on
100,000 (seed 11). (Analysis fit on the trained network.)"""
import numpy as np
from memlib import run, task, lists, score, kinds

N = 15
# (1) lone records
Xl = lists(41, 400_000); Xl = np.where(np.arange(10)[None] == np.random.default_rng(1).integers(0, 10, len(Xl))[:, None], np.abs(Xl), -np.abs(Xl))
Vl = task(Xl); _, Pl, _ = run(Xl)
print("(1) lone record (one positive number): neuron 15's pre-activation per unit of the record, median, rows step 0-9, columns age 0-9")
for t in range(10):
    cells = []
    for a in range(0, 10):
        m = Vl["age"][:, t] == a
        cells.append(f"{np.median(Pl[m, t, N] / Vl['m1'][m, t]):6.2f}" if m.sum() > 100 else "     .")
    print(f"  step {t}: " + " ".join(cells))
print("   share off, same layout:")
for t in range(10):
    cells = []
    for a in range(0, 10):
        m = Vl["age"][:, t] == a
        cells.append(f"{(Pl[m, t, N] <= 0).mean():6.2f}" if m.sum() > 100 else "     .")
    print(f"  step {t}: " + " ".join(cells))


def prevlargest(X):
    n, T = X.shape; P = np.zeros((n, T)); m1 = np.zeros(n); p = np.zeros(n)
    for t in range(T):
        x = X[:, t]; new = x > m1; p = np.where(new, m1, p); m1 = np.where(new, x, m1); P[:, t] = p
    return P


def F(X, V, t):
    x = X[:, t]; m1 = V["m1"][:, t]; m2 = V["m2"][:, t]; p = prevlargest(X)[:, t]
    return np.c_[m1, x, np.maximum(x, 0), m2, p, t * m1, t * m2, np.ones(len(X))]


Xf = lists(12, 200_000); Vf = task(Xf); _, Pf, _ = run(Xf)
X = lists(11, 100_000); V = task(X); K = kinds(X); S0, P0, _ = run(X); base = score(X, S0, S0, K)
coef = {}
for a in range(-1, 10):
    rows = [(F(Xf, Vf, t)[Vf["age"][:, t] == a], Pf[Vf["age"][:, t] == a, t, N]) for t in range(10)]
    A_ = np.concatenate([r[0] for r in rows]); b = np.concatenate([r[1] for r in rows])
    if len(b) > 100: coef[a] = np.linalg.lstsq(A_, b, rcond=None)[0]
off = np.zeros((len(X), 10), bool)
for t in range(10):
    Ft = F(X, V, t)
    for a, c in coef.items():
        m = V["age"][:, t] == a; off[m, t] = Ft[m] @ c <= 0
print(f"\n(2) compact rule: agreement with the true sign {np.mean(off == (P0[:, :, N] <= 0)):.3f} (always on {np.mean(P0[:, :, N] > 0):.3f})")
print("   coefficients by age class (m1, x, max(x,0), m2, p, t*m1, t*m2, const):")
for a, c in coef.items(): print(f"   age {a:2d}: " + " ".join(f"{v:6.2f}" for v in c))


def hook(t, pre, h):
    h = h.copy(); h[:, N] = np.where(off[:, t], 0.0, pre[:, N]); return h


S = run(X, hook)[0]; r = score(X, S0, S, K)
print(f"   neuron 15 by the compact rule, all else by own sign: acc {100 * r['all'][0]:.1f} ({100 * (r['all'][0] - base['all'][0]):+.1f}); always on: see switching.log (-7.3)")

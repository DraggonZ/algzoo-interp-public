"""Question: can each memory neuron's on/off be given by a rule written in the list's numbers, good enough that the
linear machine with these rules loses under 2 points? Rule = sign of a linear fit of the neuron's pre-activation,
fitted separately in each (step, age of the largest) cell on features of the list so far:
  'values'  : x_s and max(x_s, 0) for every position s <= t, the largest so far, the previous largest, the second-largest.
Fitted on 100,000 lists (seed 12), tested on 100,000 others (seed 11). Rules replace the sign for the listed
neurons; other memory neurons by the age table or their own sign as stated; closed loop.
(Analysis fit on the trained network; no hand-built weight comes from it.)"""
import numpy as np
from memlib import MEM, run, task, lists, score, kinds

X = lists(11, 100_000); K = kinds(X); V = task(X); age = V["age"]
S0, P0, H0 = run(X); base = score(X, S0, S0, K)
Xc = lists(12, 100_000); Vc = task(Xc); _, Pc, _ = run(Xc)


def feats(X, V, t):
    n = len(X); x = X[:, :t + 1]
    return np.c_[x, np.maximum(x, 0), V["m1"][:, t], V["m1prev"][:, t], V["m2"][:, t], np.ones(n)]


coef = {}
for t in range(10):
    Ff = feats(Xc, Vc, t)
    for a in range(-1, t + 1):
        m = Vc["age"][:, t] == a
        if m.sum() < 50: continue
        c, *_ = np.linalg.lstsq(Ff[m], Pc[m, t][:, MEM], rcond=None); coef[t, a] = c
off_rule = np.zeros((len(X), 10, 16), bool)
for t in range(10):
    F = feats(X, V, t)
    for a in range(-1, t + 1):
        m = age[:, t] == a
        if (t, a) in coef: off_rule[m, t][:, MEM] = 0; off_rule[np.ix_(np.where(m)[0], [t], MEM)] = (F[m] @ coef[t, a] <= 0)[:, None, :]
true_off = P0 <= 0
print("on/off agreement of the fitted-sign rule with the true sign (test lists), and the always-on baseline")
for n in MEM:
    print(f"  neuron {n:2d}: rule {np.mean(off_rule[:, :, n] == true_off[:, :, n]):.3f}  always on {np.mean(~true_off[:, :, n]):.3f}")


def by_rule(ns):
    def hook(t, pre, h):
        h = h.copy()
        for n in ns: h[:, n] = np.where(off_rule[:, t, n], 0.0, pre[:, n])
        return h
    return hook


print("accuracy, listed neurons by the fitted-sign rule, the rest by their own sign (closed loop)")
for ns in [[n] for n in MEM] + [[3, 15], [0, 10, 11, 12, 13, 14], MEM]:
    S = run(X, by_rule(ns))[0]; r = score(X, S0, S, K)
    print(f"  {str(ns):28s} acc {100 * r['all'][0]:.1f} ({100 * (r['all'][0] - base['all'][0]):+.1f})  " +
          " ".join(f"{k[:16]} {100 * (r[k][0] - base[k][0]):+.1f}" for k in list(K['table2b']) + list(K['source'])[:2]))

# combined: age table for some neurons, fitted-sign rule for others (closed, hybrid and open loop)
Pt = Pc  # table from the fitting sample
table = np.zeros((10, 11, 16), bool)
for t in range(10):
    for a in range(-1, 10):
        m = Vc["age"][:, t] == a
        if m.sum() > 20: table[t, a + 1] = (Pt[m, t] <= 0).mean(0) > 0.5
tab_off = table[np.arange(10)[None, :], age + 1]
from memlib import WH, WI, WO, OTH, fmt


def machine(fit_ns, tab_ns, mode):
    n = len(X); h = np.zeros((n, 16))
    for t in range(10):
        src = h if mode == "closed" else (H0[:, t - 1] if t else np.zeros((n, 16)))
        pre = h @ WH.T + X[:, t:t + 1] * WI
        pre_m = src[:, OTH] @ WH[np.ix_(MEM, OTH)].T + h[:, MEM] @ WH[np.ix_(MEM, MEM)].T + X[:, t:t + 1] * WI[MEM]
        new = np.maximum(pre, 0); off = np.zeros((n, 8), bool)
        for j, nn in enumerate(MEM):
            off[:, j] = off_rule[:, t, nn] if nn in fit_ns else tab_off[:, t, nn]
        new[:, MEM] = np.where(off, 0.0, pre_m)
        if mode == "open": new[:, OTH] = H0[:, t][:, OTH]
        h = new
    return h @ WO.T


print("\nmachine with all eight memory neurons switched by rules: fitted sign for the first set, age table for the second")
for fit_ns, tab_ns in [([0, 3, 14, 15], [10, 11, 12, 13]), ([3, 15], [0, 10, 11, 12, 13, 14]), ([0, 3, 15], [10, 11, 12, 13, 14]), ([3, 14, 15], [0, 10, 11, 12, 13])]:
    for mode in ["closed", "hybrid", "open"]:
        S = machine(fit_ns, tab_ns, mode); r = score(X, S0, S, K)
        print(f"== fitted {str(fit_ns):16s} table {str(tab_ns):22s} {mode:6s}: acc {100 * r['all'][0]:.1f} ({100 * (r['all'][0] - base['all'][0]):+.1f}) agree {100 * r['all'][1]:.1f} KL {r['all'][2]:.3f}")
        if mode == "closed": print(fmt(r, base))

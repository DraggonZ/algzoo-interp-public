"""Question (follows sign_rules.py, where single-neuron rules cost 0.0-1.2 points but all eight together 3.3): with
richer list features and fixed cells, how close does the linear machine with rule-based switching get?

Rule for neuron n at step t, in the cell (t, age of the largest):
  - if, on the fitting lists, n is off in under 3% of the cell's lists: on; over 97%: off (fixed cells);
  - otherwise: off when a linear fit of its pre-activation in that cell is <= 0. Features, for each position s <= t:
    x_s, max(x_s, 0), max(x_s, 0) if x_s was a new largest, max(x_s, 0) if the keep gate kept x_s (neuron 5 on at s+1,
    read from the real run); plus the largest so far, the previous largest, the second-largest, a constant.
Fitted on 100,000 lists (seed 12), tested on 100,000 (seed 11). Machine connected closed, hybrid and open (machine_sub.py).
(Analysis fit on the trained network; no hand-built weight comes from it.)"""
import numpy as np
from memlib import MEM, OTH, WH, WI, WO, run, task, lists, score, kinds, fmt


def feats(X, V, H, t):
    n = len(X); x = X[:, :t + 1]; rx = np.maximum(x, 0)
    kept = np.c_[H[:, 1:t + 1, 5] > 0, np.zeros((n, 1), bool)] if t < 9 else np.c_[H[:, 1:, 5] > 0, np.zeros((n, 1), bool)]
    kept = kept[:, :t + 1]
    return np.c_[x, rx, rx * V["rec"][:, :t + 1], rx * kept, V["m1"][:, t], V["m1prev"][:, t], V["m2"][:, t], np.ones(n)]


def build_rules(Xf, lo=0.03, hi=0.97):
    Vf = task(Xf); _, Pf, Hf = run(Xf); R = {}
    for t in range(10):
        F = feats(Xf, Vf, Hf, t)
        for a in range(-1, t + 1):
            m = Vf["age"][:, t] == a
            if m.sum() < 50: continue
            offr = (Pf[m, t][:, MEM] <= 0).mean(0)
            c, *_ = np.linalg.lstsq(F[m], Pf[m, t][:, MEM], rcond=None)
            R[t, a] = (offr, c)
    return R


def apply_rules(R, X, V, H, lo=0.03, hi=0.97):
    off = np.zeros((len(X), 10, 8), bool)
    for t in range(10):
        F = feats(X, V, H, t)
        for a in range(-1, t + 1):
            m = V["age"][:, t] == a
            if (t, a) not in R or not m.any(): continue
            offr, c = R[t, a]; o = F[m] @ c <= 0
            o[:, offr < lo] = False; o[:, offr > hi] = True; off[m, t] = o
    return off


def machine(X, H0, off, mode, use=None):
    """use: memory neurons switched by the rule; the others by their own sign."""
    n = len(X); h = np.zeros((n, 16)); use = MEM if use is None else use
    for t in range(10):
        src = h if mode == "closed" else (H0[:, t - 1] if t else np.zeros((n, 16)))
        pre = h @ WH.T + X[:, t:t + 1] * WI
        pre_m = src[:, OTH] @ WH[np.ix_(MEM, OTH)].T + h[:, MEM] @ WH[np.ix_(MEM, MEM)].T + X[:, t:t + 1] * WI[MEM]
        new = np.maximum(pre, 0); o = off[:, t].copy()
        for j, nn in enumerate(MEM):
            if nn not in use: o[:, j] = pre_m[:, j] <= 0
        new[:, MEM] = np.where(o, 0.0, pre_m)
        if mode == "open": new[:, OTH] = H0[:, t][:, OTH]
        h = new
    return h @ WO.T


if __name__ == "__main__":
    X = lists(11, 100_000); K = kinds(X); V = task(X)
    S0, P0, H0 = run(X); base = score(X, S0, S0, K)
    R = build_rules(lists(12, 100_000))
    off = apply_rules(R, X, V, H0); true_off = P0[:, :, MEM] <= 0
    print("on/off agreement with the true sign (test lists): " + " ".join(f"{n}:{np.mean(off[:, :, j] == true_off[:, :, j]):.3f}" for j, n in enumerate(MEM)))
    print("single neurons by the rule, the rest by their own sign (closed): " + " ".join(
        f"{n}:{100 * (score(X, S0, machine(X, H0, off, 'closed', [n]))['all'][0] - base['all'][0]):+.1f}" for n in MEM))
    for mode in ["closed", "hybrid", "open"]:
        S = machine(X, H0, off, mode); r = score(X, S0, S, K)
        print(f"== all eight by the rule, {mode}: acc {100 * r['all'][0]:.1f} ({100 * (r['all'][0] - base['all'][0]):+.1f}) agree {100 * r['all'][1]:.1f} KL {r['all'][2]:.3f}")
        print(fmt(r, base))

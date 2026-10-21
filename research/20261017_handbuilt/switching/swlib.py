"""Shared pieces for the switching study (2026-10-21). Builds on ../memory/memlib.py.

machine(X, rule): the memory as the linear machine m_t = D_t (A m_{t-1} + B o_{t-1} + w x_t), closed loop: the other
eight neurons run on their own weights and read the machine; D_t comes from rule(t, ctx), which returns a (B, 8) bool
array "off" for the memory neurons in the order 0, 3, 10-15. ctx holds the machine's pre-activations ("pre_m"), its
full state before the step ("h"), and the step. Returns final scores and the (B, 10, 8) off record.
order_keys(X): per list and step, the ages (steps since arrival) of the three largest positive numbers so far
(-1 when fewer are positive), and whether each was a record when it arrived."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "memory"))
import numpy as np
from memlib import MEM, OTH, WI, WH, WO, A, B, WIM, RM, run, task, lists, score, kinds, fmt  # noqa: F401

TEST_SEED, FIT_SEED = 31, 32


def machine(X, rule, use=None):
    """use: memory neurons switched by the rule (default all eight); the rest by their own sign."""
    n = len(X); h = np.zeros((n, 16)); offs = np.zeros((n, 10, 8), bool)
    use = list(range(8)) if use is None else [MEM.index(u) for u in use]
    for t in range(10):
        pre = h @ WH.T + X[:, t:t + 1] * WI
        pre_m = pre[:, MEM]
        off = pre_m <= 0
        r = rule(t, {"pre_m": pre_m, "h": h})
        off[:, use] = r[:, use]
        new = np.maximum(pre, 0); new[:, MEM] = np.where(off, 0.0, pre_m)
        offs[:, t] = off; h = new
    return h @ WO.T, offs


def order_keys(X):
    """ages[:, t, k] = t - position of the (k+1)-th largest positive number among x_0..x_t (-1 if none);
    isrec[:, t, k] = that number was a new largest when it arrived."""
    n, T = X.shape; ages = np.full((n, T, 3), -1); isrec = np.zeros((n, T, 3), bool)
    m1p = np.maximum.accumulate(np.c_[np.zeros(n), X], 1)[:, :T]; rec = X > m1p
    for t in range(T):
        xs = np.where(X[:, :t + 1] > 0, X[:, :t + 1], -np.inf)
        o = np.argsort(-xs, 1)[:, :3]
        for k in range(min(3, t + 1)):
            pos = o[:, k]; ok = np.isfinite(xs[np.arange(n), pos])
            ages[:, t, k] = np.where(ok, t - pos, -1); isrec[:, t, k] = ok & rec[np.arange(n), pos]
    return ages, isrec


def report(name, X, S0, S, offs, true_off, base, K=None, nfit=None):
    r = score(X, S0, S, K)
    ag = " ".join(f"{nn}:{np.mean(offs[:, :, j] == true_off[:, :, j]):.3f}" for j, nn in enumerate(MEM))
    print(f"== {name}: acc {100 * r['all'][0]:.2f} ({100 * (r['all'][0] - base['all'][0]):+.2f}) agree-answer {100 * r['all'][1]:.1f}"
          + (f"  fitted numbers {nfit}" if nfit is not None else ""))
    print(f"   on/off agreement with the network's own run: {ag}  (all {np.mean(offs == true_off):.4f})")
    if K is not None: print(fmt(r, base))
    sys.stdout.flush()
    return r

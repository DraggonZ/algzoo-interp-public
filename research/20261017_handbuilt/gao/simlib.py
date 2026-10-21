"""Shared pieces for the simulation-score tests (Gao, "Neural networks are not already clean", 2026).

The trained network is run with some neurons' values replaced, at every step, by values simulated from a formula of the
input list alone. The score is how far the output moves: KL from the trained output distribution (softmax of the ten
scores) to the substituted one, averaged over lists, plus the accuracy and agreement with the trained answer."""
import numpy as np
from algzoo.core import example

WI, WH, WO = (np.asarray(w, dtype=float) for w in example())
WI = WI.reshape(-1)
GROUPS = {"store+detector": [2, 4], "cancellers": [1, 6, 7, 8], "keep gate": [5], "late gate": [9],
          "memory": [0, 3, 10, 11, 12, 13, 14, 15]}


def lists(seed, n=200_000):
    return np.random.default_rng(seed).standard_normal((n, 10))


def run(X, sub=None, params=None):
    """Run the trained network (or params = (wi, wh, wo) of another one). sub: {neuron: array (B, 10)} of values to force after each step's update.
    Returns final scores (B, 10), pre-activations (B, 10, 16) and activations (B, 10, 16)."""
    wi, wh, wo = params if params is not None else (WI, WH, WO)
    B = len(X); h = np.zeros((B, len(wi))); P, H = [], []
    for t in range(10):
        pre = h @ wh.T + X[:, t:t + 1] * wi; h = np.maximum(pre, 0)
        if sub:
            for n, v in sub.items():
                h[:, n] = v[:, t]
        P.append(pre); H.append(h.copy())
    return h @ wo.T, np.stack(P, 1), np.stack(H, 1)


def task_vars(X):
    """Quantities the account speaks of, per list and step t (after input t). 'before' means over inputs 0..t-1,
    with zeros padded so they are 0 at the start (as the fits in cancellers/arc_canceller_code.py)."""
    B = len(X); V = {k: np.zeros((B, 10)) for k in ("x", "xprev", "m1", "m2", "m1in", "m2in", "jump", "age", "newmax")}
    m1 = np.zeros(B); m2 = np.zeros(B); arg = np.full(B, -1)
    for t in range(10):
        x = X[:, t]
        V["x"][:, t] = x; V["xprev"][:, t] = X[:, t - 1] if t else 0
        V["m1"][:, t] = m1; V["m2"][:, t] = m2; V["jump"][:, t] = np.maximum(x - m1, 0)
        new = x > m1; V["newmax"][:, t] = new
        m2 = np.where(new, m1, np.maximum(m2, x)); m1 = np.where(new, x, m1); arg = np.where(new, t, arg)
        V["m1in"][:, t] = m1; V["m2in"][:, t] = m2; V["age"][:, t] = np.where(arg >= 0, t - arg, -1)
    return V


def score(X, logits_true, logits_sub):
    def lsm(z):
        z = z - z.max(1, keepdims=True); return z - np.log(np.exp(z).sum(1, keepdims=True))
    lp, lq = lsm(logits_true), lsm(logits_sub)
    kl = (np.exp(lp) * (lp - lq)).sum(1).mean()
    y = X.argsort(1)[:, -2]
    return {"kl": kl, "acc": (logits_sub.argmax(1) == y).mean(), "agree": (logits_sub.argmax(1) == logits_true.argmax(1)).mean()}


def fit_relu(F_fit, pre_fit, F_score):
    """Least squares of the pre-activation on features (all lists and steps pooled), then ReLU. F: (B, 10, nf)."""
    A = F_fit.reshape(-1, F_fit.shape[-1]); co = np.linalg.lstsq(A, pre_fit.reshape(-1), rcond=None)[0]
    return np.maximum(F_score @ co, 0), co

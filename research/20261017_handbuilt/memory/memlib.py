"""Shared pieces for the memory study (2026-10-18). ARC's 16-neuron second-argmax RNN, h <- ReLU(W_hh h + W_hi x_t),
scores = W_oh h after ten steps. Memory = neurons 0, 3, 10-15 (in that order everywhere here); others = 1, 2, 4-9.

run(): runs the network, optionally with a hook that rewrites the state after each step.
task(): per list and step, the quantities the account speaks of (largest so far, age of the largest, ...).
kinds(): two ways to split lists into kinds (table 2b's exclusive kinds; where the second-largest came from).
score(): accuracy, agreement with the trained answer and KL from the trained output, overall and by kind."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
import numpy as np
from algzoo.core import example

WI, WH, WO = (np.asarray(w, dtype=float) for w in example())
WI = WI.reshape(-1)
MEM = [0, 3, 10, 11, 12, 13, 14, 15]
OTH = [1, 2, 4, 5, 6, 7, 8, 9]
NAMES = {1: "hiding 1", 6: "hiding 6", 7: "hiding 7", 8: "hiding 8", 2: "store 2", 4: "detector 4", 5: "keep gate 5", 9: "late gate 9"}
A = WH[np.ix_(MEM, MEM)]          # memory -> memory
B = WH[np.ix_(MEM, OTH)]          # others -> memory
WIM = WI[MEM]                     # input -> memory
RM = WO[:, MEM]                   # memory readout (10 x 8)
RO = WO[:, OTH]


def lists(seed, n):
    return np.random.default_rng(seed).standard_normal((n, 10))


def run(X, hook=None, params=None):
    """Returns final scores (B,10), pre-activations (B,10,16), activations (B,10,16).
    hook(t, pre, h) may return a new h (after the ReLU) for step t."""
    wi, wh, wo = params if params is not None else (WI, WH, WO)
    n = len(X); h = np.zeros((n, len(wi))); P, H = [], []
    for t in range(X.shape[1]):
        pre = h @ wh.T + X[:, t:t + 1] * wi; h = np.maximum(pre, 0)
        if hook is not None:
            out = hook(t, pre, h)
            if out is not None: h = out
        P.append(pre); H.append(h.copy())
    return h @ wo.T, np.stack(P, 1), np.stack(H, 1)


def task(X):
    """m1: largest so far after step t (0 if none positive); m1prev: before step t; age: t - position of the largest
    so far (-1 while no positive number); rec: x_t is a new largest (> all earlier and > 0); m2: second-largest so far."""
    n, T = X.shape; V = {k: np.zeros((n, T)) for k in ("m1", "m1prev", "m2", "age", "rec", "jump")}
    m1 = np.zeros(n); m2 = np.zeros(n); arg = np.full(n, -1)
    for t in range(T):
        x = X[:, t]; new = x > m1
        V["m1prev"][:, t] = m1; V["jump"][:, t] = np.maximum(x - m1, 0); V["rec"][:, t] = new
        m2 = np.where(new, m1, np.maximum(m2, x)); m1 = np.where(new, x, m1); arg = np.where(new, t, arg)
        V["m1"][:, t] = m1; V["m2"][:, t] = m2; V["age"][:, t] = np.where(arg >= 0, t - arg, -1)
    V["rec"] = V["rec"].astype(bool); V["age"] = V["age"].astype(int)
    return V


def kinds(X):
    """Two splits. 'table2b': each list goes to the first kind it matches (at most one positive; new largest at step 9;
    at step 8; two new largests in a row earlier; ordinary), as cancellers/table2b_exclusive.log.
    'source': where the second-largest came from: after the largest (a kept vote), an overtaken record (revealed),
    before the largest and never a record."""
    n, T = X.shape
    m1p = np.stack([np.max(np.c_[np.zeros((n, 2)), X[:, :t]], 1) for t in range(T)], 1); rec = X > m1p
    cons = (rec[:, :-1] & rec[:, 1:]).any(1); onepos = (X > 0).sum(1) <= 1
    a = {"at most one positive": onepos}
    a["new largest at step 9"] = rec[:, 9] & ~onepos
    a["new largest at step 8"] = rec[:, 8] & ~rec[:, 9] & ~onepos
    a["two in a row, earlier"] = cons & ~rec[:, 8] & ~rec[:, 9] & ~onepos
    a["ordinary"] = ~onepos & ~rec[:, 9] & ~rec[:, 8] & ~cons
    o = X.argsort(1); s, l = o[:, -2], o[:, -1]; srec = rec[np.arange(n), s]
    b = {"second after largest": s > l, "second an overtaken record": (s < l) & srec, "second before largest, not a record": (s < l) & ~srec}
    return {"table2b": a, "source": b}


def lsm(z):
    z = z - z.max(1, keepdims=True); return z - np.log(np.exp(z).sum(1, keepdims=True))


def score(X, S_true, S_sub, K=None):
    """Accuracy (names the second-largest), agreement with the trained answer, KL(trained || substituted); by kind."""
    y = X.argsort(1)[:, -2]; ok = S_sub.argmax(1) == y; ag = S_sub.argmax(1) == S_true.argmax(1)
    lp, lq = lsm(S_true), lsm(S_sub); kl = (np.exp(lp) * (lp - lq)).sum(1)
    out = {"all": (ok.mean(), ag.mean(), kl.mean(), len(X))}
    if K is not None:
        for split in K.values():
            for k, m in split.items():
                out[k] = (ok[m].mean() if m.any() else np.nan, ag[m].mean() if m.any() else np.nan, kl[m].mean() if m.any() else np.nan, int(m.sum()))
    return out


def fmt(res, base=None):
    """One line per kind: accuracy (and its change from base), agreement, KL, share."""
    n = res["all"][3]; lines = []
    for k, (acc, ag, kl, c) in res.items():
        d = f" ({100 * (acc - base[k][0]):+5.1f})" if base is not None else ""
        lines.append(f"    {k:38s} acc {100 * acc:5.1f}{d}  agree {100 * ag:5.1f}  KL {kl:6.3f}  share {c / n:5.3f}")
    return "\n".join(lines)

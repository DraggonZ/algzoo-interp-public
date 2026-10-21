"""r1b, no waves: the constrained lean rule without the timer-wave features (cos, sin of the largest's age at each
neuron's angle). Mechanism tested: the timer waves are produced by the hiding neurons' own mutual feedback (the rule
runs on itself), so they need not be read from outside. Structure: 7 feature choices x 4 neurons = 28 bits; no
hand-set angles. 11 free constants per neuron (7 features + 4 feedback).

Derived from: the constrained lean hiding-neuron rule (README experiment 24, step18_constrained.py), as a rule module for
../score_rule.py. Input to hiding neuron j at step t:
    w_in,j x_t + sum_f c_j,f feature_f + sum_k b_j,k (h_k(t-1) - w_in,k x_{t-1})
with ARC's input weights, the rule's own previous hiding values made blind to the previous number, and 9 features:
largest, second-largest, timer wave (cos, sin) at the neuron's angle, store and detector from their recursion,
previous jump, jump two steps back, new largest's value two steps back. 13 free constants per neuron."""
import numpy as np
import torch as th
from common import HID, WI, account_features, sd_rule, task_vars, lag

dev = "cuda"
NF = 7
KINDS = np.array([["feature"] * NF + ["feedback"] * 4] * 4)          # one spread per kind
# which 9 features (1 bit each per neuron) + the 4 timer angles at 6 bits each (hand-set, from the account)
STRUCTURE_BITS = 7 * 4
win = th.tensor(WI[HID], device=dev)


def feats(X, n):
    V = task_vars(X); A = account_features(X, n); sd = sd_rule(X)
    F = [A[..., 1], A[..., 2], sd[..., 0], sd[..., 1],
         lag(V["jump"]), lag(V["jump"], 2), lag(V["newmax"] * V["x"], 2)]
    return np.stack(F, -1)


def prepare(X):
    return {"F": th.tensor(np.stack([feats(X, n) for n in HID], 2), device=dev),   # (B, 10, 4, 9)
            "x": th.tensor(X, device=dev)}


def rollout(d, C):
    F, x = d["F"], d["x"]
    prev = th.zeros(F.shape[0], 4, device=dev, dtype=F.dtype); pres = []
    for t in range(10):
        xp = x[:, t - 1:t] if t else th.zeros_like(x[:, :1])
        pre = win * x[:, t:t + 1] + (F[:, t] * C[None, :, :NF]).sum(-1) + (prev - xp * win) @ C[:, NF:].T
        pres.append(pre); prev = th.relu(pre)
    return th.stack(pres, 1)


def init_constants(X, P, H):
    """One-step least squares with ARC's previous hiding values (analysis fit on ARC's activity)."""
    Fn = np.stack([feats(X, n) for n in HID], 2)
    Hp = np.concatenate([np.zeros((len(X), 1, 16)), H[:, :-1]], 1)[..., HID]
    xprev = lag(X)
    blindfb = Hp - xprev[..., None] * WI[HID]
    C0 = []
    for i, n in enumerate(HID):
        A = np.concatenate([Fn[:, :, i], blindfb], -1)
        C0.append(np.linalg.lstsq(A.reshape(-1, A.shape[-1]), (P[:, :, n] - WI[n] * X).reshape(-1), rcond=None)[0])
    return np.stack(C0)

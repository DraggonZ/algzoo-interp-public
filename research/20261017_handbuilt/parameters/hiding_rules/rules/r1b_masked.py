"""r1b, masked: the store-reference rule (r1b_store_ref.py) with the record-aftermath terms removed where they do not
reach. Input to hiding neuron j at step t:
    w_in,j x_t + sum_f c_j,f feature_f + sum_k b_j,k (h_k(t-1) - w_in,k x_{t-1})
with ARC's input weights and the rule's own previous hiding values made blind to the previous number.

Mechanisms stated:
(1) the timer waves come from the hiding neurons' mutual feedback, not from outside: no wave features, no angles;
(2) the hiding neurons get the largest so far only through the store and detector (neurons 2 and 4, from their own
    2x2 recursion with ARC's weights): no separate largest-so-far feature;
(3) the aftermath of a record (previous jump; the late gate's push two steps later, as jump and new largest) does not
    reach neuron 7, and the late gate's new-largest push does not reach neuron 8.
Features per neuron: second-largest, store, detector for all four; previous jump, jump two back for 1, 6, 8; new
largest two back for 1 and 6. 20 feature constants + 16 feedback constants = 36.

Structure: one bit per (neuron, feature) choice over the 6 candidate features, 6 x 4 = 24 bits. The masks are those
choices, so they cost nothing extra. No hand-set numbers."""
import numpy as np
import torch as th
from common import HID, WI, sd_rule, task_vars, lag

dev = "cuda"
NF = 6
FEATS = ["m2", "store", "detector", "prev jump", "jump 2 back", "new largest 2 back"]
MASK = np.array([[1, 1, 1, 1, 1, 1],      # neuron 1
                 [1, 1, 1, 1, 1, 1],      # neuron 6
                 [1, 1, 1, 0, 0, 0],      # neuron 7: no record aftermath
                 [1, 1, 1, 1, 1, 0]], bool)   # neuron 8: no new-largest push
FULL = np.concatenate([MASK, np.ones((4, 4), bool)], 1)     # (4, NF + 4): which entries are free constants
KINDS = np.array([("feature" if f < NF else "feedback") for i in range(4) for f in range(NF + 4) if FULL[i, f]])
STRUCTURE_BITS = NF * 4
win = th.tensor(WI[HID], device=dev)
_idx = th.tensor(np.flatnonzero(FULL.ravel()), device=dev)


def feats(X):
    V = task_vars(X); sd = sd_rule(X)
    return np.stack([V["m2"], sd[..., 0], sd[..., 1], lag(V["jump"]), lag(V["jump"], 2), lag(V["newmax"] * V["x"], 2)], -1)


def prepare(X):
    return {"F": th.tensor(feats(X), device=dev), "x": th.tensor(X, device=dev)}      # (B, 10, 6), shared by all four


def full(C):
    M = th.zeros(4 * (NF + 4), device=dev, dtype=C.dtype)
    return M.index_put((_idx,), C.reshape(-1)).view(4, NF + 4)


def rollout(d, C):
    F, x = d["F"], d["x"]; M = full(C)
    prev = th.zeros(F.shape[0], 4, device=dev, dtype=F.dtype); pres = []
    for t in range(10):
        xp = x[:, t - 1:t] if t else th.zeros_like(x[:, :1])
        pre = win * x[:, t:t + 1] + F[:, t] @ M[:, :NF].T + (prev - xp * win) @ M[:, NF:].T
        pres.append(pre); prev = th.relu(pre)
    return th.stack(pres, 1)


def init_constants(X, P, H):
    """One-step least squares with ARC's previous hiding values (analysis fit on ARC's activity)."""
    Fn = feats(X)
    Hp = np.concatenate([np.zeros((len(X), 1, 16)), H[:, :-1]], 1)[..., HID]
    blindfb = Hp - lag(X)[..., None] * WI[HID]
    C0 = []
    for i, n in enumerate(HID):
        A = np.concatenate([Fn[..., MASK[i]], blindfb], -1)
        C0.append(np.linalg.lstsq(A.reshape(-1, A.shape[-1]), (P[:, :, n] - WI[n] * X).reshape(-1), rcond=None)[0])
    return np.concatenate(C0)

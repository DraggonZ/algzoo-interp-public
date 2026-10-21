"""r2b, tied: the masked rule (r1b_masked.py) where hiding neurons that read a feature alike share one coefficient.
Input to hiding neuron j at step t:
    w_in,j x_t + sum_f c_j,f feature_f + sum_k b_j,k (h_k(t-1) - w_in,k x_{t-1})
with ARC's input weights and the rule's own previous hiding values made blind to the previous number.

Mechanisms stated (1-3 as r1b_masked.py):
(1) the timer waves come from the hiding neurons' mutual feedback: no wave features, no angles;
(2) the largest so far reaches the hiding neurons only through the store and detector (their own 2x2 recursion with
    ARC's weights);
(3) the aftermath of a record does not reach neuron 7, and the late gate's new-largest push does not reach neuron 8;
(4) the four hiding neurons take the same share of the store as their reference (as r2b_shared_store.py);
(5) the hiding neurons come in two pairs by how they read the second-largest: 1 and 6 add it with one strength, 7 and 8
    subtract it with one strength;
(6) the jump two steps back pushes neurons 1, 6 and 8 by the same amount.
One-step checks (r2b_shared.log, r2b_shared_more.log): each tie costs under 0.003 rms.
Constants: 20 feature constants - 3 (store) - 2 (second-largest pairs) - 2 (jump two back) = 13, + 16 feedback = 29.

Structure: one bit per (neuron, feature) choice over the 6 candidate features, 6 x 4 = 24 bits; per tie, 1 bit for
tying a column; for the second-largest also log2(3) bits for which of the 3 pairings of four neurons: 1 + 3 + 1 = 5
(rounded up). Total 29. No hand-set numbers. Caveat: the ties were found by inspecting the fitted constants, so this
count may undercharge for the search over possible ties."""
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
TIES = [(1, [0, 1, 2, 3]), (0, [0, 1]), (0, [2, 3]), (4, [0, 1, 3])]     # (feature column, neuron indices sharing it)
FULL = np.concatenate([MASK, np.ones((4, 4), bool)], 1)
OWN = FULL.copy()
for col, who in TIES:
    OWN[who, col] = False
NOWN = int(OWN.sum()); NT = len(TIES)
SRC = np.full(4 * (NF + 4), NT + NOWN)                  # index into [C, 0]; C = [tied constants, own constants]
SRC[np.flatnonzero(OWN.ravel())] = NT + np.arange(NOWN)
for t, (col, who) in enumerate(TIES):
    SRC[[i * (NF + 4) + col for i in who]] = t
KINDS = np.array(["feature"] * NT + [("feature" if f < NF else "feedback") for i in range(4) for f in range(NF + 4) if OWN[i, f]])
STRUCTURE_BITS = NF * 4 + 5
win = th.tensor(WI[HID], device=dev)
_src = th.tensor(SRC, device=dev)


def feats(X):
    V = task_vars(X); sd = sd_rule(X)
    return np.stack([V["m2"], sd[..., 0], sd[..., 1], lag(V["jump"]), lag(V["jump"], 2), lag(V["newmax"] * V["x"], 2)], -1)


def prepare(X):
    return {"F": th.tensor(feats(X), device=dev), "x": th.tensor(X, device=dev)}


def full(C):
    C = C.reshape(-1)
    return th.cat([C, C.new_zeros(1)])[_src].view(4, NF + 4)


def rollout(d, C):
    F, x = d["F"], d["x"]; M = full(C)
    prev = th.zeros(F.shape[0], 4, device=dev, dtype=F.dtype); pres = []
    for t in range(10):
        xp = x[:, t - 1:t] if t else th.zeros_like(x[:, :1])
        pre = win * x[:, t:t + 1] + F[:, t] @ M[:, :NF].T + (prev - xp * win) @ M[:, NF:].T
        pres.append(pre); prev = th.relu(pre)
    return th.stack(pres, 1)


def init_constants(X, P, H):
    """One-step least squares per neuron with ARC's previous hiding values (analysis fit on ARC's activity); each tied
    constant starts at the mean of the free values it replaces. The scorer's fit then refines all constants."""
    Fn = feats(X)
    Hp = np.concatenate([np.zeros((len(X), 1, 16)), H[:, :-1]], 1)[..., HID]
    blindfb = Hp - lag(X)[..., None] * WI[HID]
    M = np.zeros((4, NF + 4))
    for i, n in enumerate(HID):
        A = np.concatenate([Fn[..., MASK[i]], blindfb], -1)
        on = (P[:, :, n] > 0).reshape(-1)
        c = np.linalg.lstsq(A.reshape(-1, A.shape[-1])[on], (P[:, :, n] - WI[n] * X).reshape(-1)[on], rcond=None)[0]
        M[i, FULL[i]] = c
    return np.concatenate([[M[who, col].mean() for col, who in TIES], M[OWN]])

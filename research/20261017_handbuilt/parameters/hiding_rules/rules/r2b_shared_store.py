"""r2b, shared store: the masked rule (r1b_masked.py) with one store coefficient for all four hiding neurons.
Input to hiding neuron j at step t:
    w_in,j x_t + a * store + sum_f c_j,f feature_f + sum_k b_j,k (h_k(t-1) - w_in,k x_{t-1})
with ARC's input weights and the rule's own previous hiding values made blind to the previous number.

Mechanisms stated (1-3 as r1b_masked.py):
(1) the timer waves come from the hiding neurons' mutual feedback: no wave features, no angles;
(2) the largest so far reaches the hiding neurons only through the store and detector (neurons 2 and 4, their own 2x2
    recursion with ARC's weights);
(3) the aftermath of a record does not reach neuron 7, and the late gate's new-largest push does not reach neuron 8;
(4) new: the four hiding neurons take the same share a of the store as their reference. The reference built from the
    largest so far is common to the group; they differ in how they read the second-largest, the detector and each
    other. (One-step check r2b_shared.py: a shared value costs under 0.002 rms. The detector cannot be shared.)
Constants: 1 shared store + 16 other feature constants + 16 feedback constants = 33.

Structure: one bit per (neuron, feature) choice over the 6 candidate features, 6 x 4 = 24 bits, plus 1 bit for the
choice to share the store coefficient = 25. No hand-set numbers."""
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
FULL = np.concatenate([MASK, np.ones((4, 4), bool)], 1)     # (4, NF + 4): which entries are nonzero
OWN = FULL.copy(); OWN[:, 1] = False                        # entries with their own constant (store is shared)
NOWN = int(OWN.sum())
# C = [shared store, own constants in row-major order]; SRC maps each of the 4 x (NF+4) entries to an index into
# [C, 0] (index len(C) = the zero slot)
SRC = np.full(4 * (NF + 4), 1 + NOWN)
SRC[np.flatnonzero(OWN.ravel())] = 1 + np.arange(NOWN)
SRC[[i * (NF + 4) + 1 for i in range(4)]] = 0
KINDS = np.array(["feature"] + [("feature" if f < NF else "feedback") for i in range(4) for f in range(NF + 4) if OWN[i, f]])
STRUCTURE_BITS = NF * 4 + 1
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
    """One-step least squares with ARC's previous hiding values (analysis fit on ARC's activity), per neuron; the shared
    store coefficient starts at the mean of the four free values. The scorer's fit then refines all constants."""
    Fn = feats(X)
    Hp = np.concatenate([np.zeros((len(X), 1, 16)), H[:, :-1]], 1)[..., HID]
    blindfb = Hp - lag(X)[..., None] * WI[HID]
    M = np.zeros((4, NF + 4))
    for i, n in enumerate(HID):
        A = np.concatenate([Fn[..., MASK[i]], blindfb], -1)
        on = (P[:, :, n] > 0).reshape(-1)
        c = np.linalg.lstsq(A.reshape(-1, A.shape[-1])[on], (P[:, :, n] - WI[n] * X).reshape(-1)[on], rcond=None)[0]
        M[i, FULL[i]] = c
    return np.concatenate([[M[:, 1].mean()], M[OWN]])

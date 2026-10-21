"""r3b, closed loop: r2b_shared_store.py run inside ARC's network.
Input to hiding neuron j at step t:
    w_in,j x_t + a * h_2(t-1) + sum_f c_j,f feature_f + sum_k b_j,k (h_k(t-1) - w_in,k x_{t-1})
where h_2 and h_4 (store and detector) are the network's own neurons. The 12 non-hiding neurons run with ARC's own
rows (input weight and 16 recurrent weights), which the description states anyway; their inputs from the hiding
neurons come from the rule's hiding activity. So the store and detector hear the hiding and memory neurons, as in
ARC, instead of only each other (the 2x2 recursion of earlier rules).

Mechanisms stated (1-4 as r2b_shared_store.py):
(1) the timer waves come from the hiding neurons' mutual feedback: no wave features, no angles;
(2) the largest so far reaches the hiding neurons only through the store and detector;
(3) the aftermath of a record does not reach neuron 7, and the late gate's new-largest push does not reach neuron 8;
(4) the four hiding neurons take the same share a of the store;
(5) new: the store and detector they read are the network's own neurons 2 and 4, driven by all their inputs.
Constants: 1 shared store + 16 other feature constants + 16 feedback constants = 33.

Structure: one bit per (neuron, feature) choice over the 6 candidate features, 6 x 4 = 24 bits, plus 1 bit for the
shared store = 25. No hand-set numbers. Running the other neurons with ARC's rows adds no number."""
import numpy as np
import torch as th
from common import HID, WI, WH, task_vars, lag

dev = "cuda"
NF = 6
FEATS = ["m2", "store", "detector", "prev jump", "jump 2 back", "new largest 2 back"]
LIST_COLS = [0, 3, 4, 5]                  # features that are list quantities; 1 and 2 come from the network
MASK = np.array([[1, 1, 1, 1, 1, 1],      # neuron 1
                 [1, 1, 1, 1, 1, 1],      # neuron 6
                 [1, 1, 1, 0, 0, 0],      # neuron 7: no record aftermath
                 [1, 1, 1, 1, 1, 0]], bool)   # neuron 8: no new-largest push
FULL = np.concatenate([MASK, np.ones((4, 4), bool)], 1)
OWN = FULL.copy(); OWN[:, 1] = False
NOWN = int(OWN.sum())
SRC = np.full(4 * (NF + 4), 1 + NOWN)
SRC[np.flatnonzero(OWN.ravel())] = 1 + np.arange(NOWN)
SRC[[i * (NF + 4) + 1 for i in range(4)]] = 0
KINDS = np.array(["feature"] + [("feature" if f < NF else "feedback") for i in range(4) for f in range(NF + 4) if OWN[i, f]])
STRUCTURE_BITS = NF * 4 + 1
OTHER = [n for n in range(16) if n not in HID]
win = th.tensor(WI[HID], device=dev)
_src = th.tensor(SRC, device=dev)
WH_o = th.tensor(WH[OTHER], device=dev)        # (12, 16) ARC's rows of the other neurons
WI_o = th.tensor(WI[OTHER], device=dev)
_hid = th.tensor(HID, device=dev); _oth = th.tensor(OTHER, device=dev)


def list_feats(X):
    V = task_vars(X)
    return np.stack([V["m2"], lag(V["jump"]), lag(V["jump"], 2), lag(V["newmax"] * V["x"], 2)], -1)


def prepare(X):
    return {"F": th.tensor(list_feats(X), device=dev), "x": th.tensor(X, device=dev)}


def full(C):
    C = C.reshape(-1)
    return th.cat([C, C.new_zeros(1)])[_src].view(4, NF + 4)


def rollout(d, C):
    F, x = d["F"], d["x"]; M = full(C)
    B = F.shape[0]
    h = th.zeros(B, 16, device=dev, dtype=F.dtype); pres = []
    Ml = M[:, LIST_COLS]; Msd = M[:, 1:3]; Mfb = M[:, NF:]
    for t in range(10):
        xt = x[:, t:t + 1]
        xp = x[:, t - 1:t] if t else th.zeros_like(xt)
        hp = h[:, _hid]
        pre = win * xt + F[:, t] @ Ml.T + h[:, [2, 4]] @ Msd.T + (hp - xp * win) @ Mfb.T
        oth = th.relu(h @ WH_o.T + xt * WI_o)
        h = th.zeros_like(h).index_copy(1, _oth, oth).index_copy(1, _hid, th.relu(pre))
        pres.append(pre)
    return th.stack(pres, 1)


def init_constants(X, P, H):
    """One-step least squares on ARC's states (ARC's own store and detector one step back), per neuron; the shared
    store coefficient starts at the mean of the four free values. The scorer's fit then refines all constants."""
    L = list_feats(X)
    Hp = np.concatenate([np.zeros((len(X), 1, 16)), H[:, :-1]], 1)
    Fn = np.stack([L[..., 0], Hp[..., 2], Hp[..., 4], L[..., 1], L[..., 2], L[..., 3]], -1)
    blindfb = Hp[..., HID] - lag(X)[..., None] * WI[HID]
    M = np.zeros((4, NF + 4))
    for i, n in enumerate(HID):
        A = np.concatenate([Fn[..., MASK[i]], blindfb], -1)
        on = (P[:, :, n] > 0).reshape(-1)
        c = np.linalg.lstsq(A.reshape(-1, A.shape[-1])[on], (P[:, :, n] - WI[n] * X).reshape(-1)[on], rcond=None)[0]
        M[i, FULL[i]] = c
    return np.concatenate([[M[:, 1].mean()], M[OWN]])
